import Foundation

/// One server-sent event from `POST /api/chat`.
///
/// The cases mirror the emit sites in `onit/ui/api.py`. `think` is deliberately
/// a separate case from `token`: the backend keeps reasoning and answer on two
/// channels so a client can fold one away without touching the other, and
/// collapsing them here would throw that away before the UI ever sees it.
enum ChatEvent: Sendable, Equatable {
    case token(String)
    case think(String)
    case phaseEnd(String)
    case status(String)
    case answerStart
    case approval(ApprovalRequest)
    case approvalClosed(String)
    case correction(Correction)
    case done(DonePayload)
    case error(String)
    /// An event name this client does not know. Kept rather than dropped so a
    /// backend that grows a new event does not silently lose it.
    case unknown(name: String, data: String)
}

/// A gated shell command waiting on a person.
///
/// `expiresIn` is the server's own budget. The card counts down from it and
/// lets it run out: the backend resolves an unanswered request to a deny, and
/// the UI must not pretend otherwise by keeping a dead card on screen.
struct ApprovalRequest: Sendable, Identifiable, Equatable {
    let id: String
    let command: String
    let reason: String
    let subjects: [String]
    let expiresIn: Double
}

struct Correction: Sendable, Equatable {
    let content: String
    let note: String
}

/// The final event of a successful run.
struct DonePayload: Sendable, Equatable {
    let content: String
    let files: [FileInfo]
    let elapsed: Double
    let tokensPerSecond: Double
    let turn: Int
    let ratingEnabled: Bool
    let timing: Timing
}

struct Timing: Sendable, Equatable {
    let turns: Int
    let toolCalls: Int
    let modelSeconds: Double
    let promptTokensTotal: Int
    let completionTokens: Int
    /// The backend's own one-line summary. Shown verbatim in the status bar
    /// rather than reassembled from the numbers, so the app and the browser
    /// never disagree about what a run cost.
    let tokenStats: String
}

struct FileInfo: Sendable, Equatable, Identifiable, Decodable {
    let name: String
    /// Path relative to the connection's base URL, already percent-encoded.
    let url: String
    let size: Int
    let kind: String
    let preview: String?
    /// The server's MIME guess for audio, so the player is fed what the file
    /// claims to be. Nil for every other kind.
    let mime: String?

    var id: String { url }

    var isImage: Bool { kind == "image" }
    var isCode: Bool { kind == "code" }
    var isAudio: Bool { kind == "audio" }

    init(
        name: String, url: String, size: Int, kind: String,
        preview: String? = nil, mime: String? = nil
    ) {
        self.name = name
        self.url = url
        self.size = size
        self.kind = kind
        self.preview = preview
        self.mime = mime
    }

    /// History decoding. Lenient on purpose: one unreadable descriptor must
    /// not cost the row every attachment, and a field of the wrong shape
    /// (a code preview arrives as an object) becomes nil rather than fatal.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            name: (try? c.decodeIfPresent(String.self, forKey: .name)) ?? "",
            url: (try? c.decodeIfPresent(String.self, forKey: .url)) ?? "",
            size: (try? c.decodeIfPresent(Int.self, forKey: .size)) ?? 0,
            kind: (try? c.decodeIfPresent(String.self, forKey: .kind)) ?? "file",
            preview: (try? c.decodeIfPresent(String.self, forKey: .preview)) ?? nil,
            mime: (try? c.decodeIfPresent(String.self, forKey: .mime)) ?? nil
        )
    }

    private enum CodingKeys: String, CodingKey {
        case name, url, size, kind, preview, mime
    }
}

// MARK: - Decoding

extension ChatEvent {
    /// Decode one `event:`/`data:` pair.
    ///
    /// Decoding is deliberately lenient: a malformed payload becomes `.error`
    /// rather than throwing, because one bad frame must not tear down a stream
    /// that is otherwise healthy.
    static func decode(name: String, data: String) -> ChatEvent {
        guard let json = JSON.object(from: data) else {
            return .unknown(name: name, data: data)
        }
        switch name {
        case "token":
            return .token(json["delta"] as? String ?? "")
        case "think":
            return .think(json["delta"] as? String ?? "")
        case "phase_end":
            return .phaseEnd(json["content"] as? String ?? "")
        case "status":
            return .status(json["text"] as? String ?? "")
        case "answer_start":
            return .answerStart
        case "approval":
            return .approval(
                ApprovalRequest(
                    id: json["approval_id"] as? String ?? "",
                    command: json["command"] as? String ?? "",
                    reason: json["reason"] as? String ?? "",
                    subjects: (json["subjects"] as? [String]) ?? [],
                    expiresIn: (json["expires_in"] as? Double)
                        ?? Double(json["expires_in"] as? Int ?? 0)
                )
            )
        case "approval_closed":
            return .approvalClosed(json["approval_id"] as? String ?? "")
        case "correction":
            return .correction(
                Correction(
                    content: json["content"] as? String ?? "",
                    note: json["note"] as? String ?? ""
                )
            )
        case "done":
            return .done(DonePayload(json))
        case "error":
            return .error(json["message"] as? String ?? "Unknown error")
        default:
            return .unknown(name: name, data: data)
        }
    }
}

extension DonePayload {
    init(_ json: [String: Any]) {
        let timingJSON = json["timing"] as? [String: Any] ?? [:]
        self.init(
            content: json["content"] as? String ?? "",
            files: (json["files"] as? [[String: Any]] ?? []).map(FileInfo.init),
            elapsed: (json["elapsed"] as? Double) ?? 0,
            tokensPerSecond: (json["tok_s"] as? Double) ?? 0,
            turn: (json["turn"] as? Int) ?? 0,
            ratingEnabled: (json["rating_enabled"] as? Bool) ?? false,
            timing: Timing(
                turns: (timingJSON["turns"] as? Int) ?? 0,
                toolCalls: (timingJSON["tool_calls"] as? Int) ?? 0,
                modelSeconds: (timingJSON["model_s"] as? Double) ?? 0,
                promptTokensTotal: (timingJSON["prompt_tokens_total"] as? Int) ?? 0,
                completionTokens: (timingJSON["completion_tokens"] as? Int) ?? 0,
                tokenStats: (timingJSON["token_stats"] as? String) ?? ""
            )
        )
    }
}

extension FileInfo {
    /// The stream path's dictionary form, as DonePayload receives it.
    init(_ json: [String: Any]) {
        self.init(
            name: json["name"] as? String ?? "",
            url: json["url"] as? String ?? "",
            size: (json["size"] as? Int) ?? 0,
            kind: json["kind"] as? String ?? "file",
            preview: json["preview"] as? String,
            mime: json["mime"] as? String
        )
    }
}

enum JSON {
    static func object(from text: String) -> [String: Any]? {
        guard let data = text.data(using: .utf8) else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }
}
