import Foundation

/// The one place that knows the OnIt HTTP contract.
///
/// Every request carries `X-Session-Id`; the server resolves the session from
/// that header (`onit/ui/api.py:779`), so nothing here depends on a cookie jar
/// and two sessions can be in flight against one connection at once.
struct OnItClient: Sendable {
    let baseURL: URL

    init(baseURL: URL) {
        self.baseURL = baseURL
    }

    // MARK: - Session model

    struct SessionSummary: Sendable, Decodable, Identifiable, Equatable {
        let sessionID: String
        let tag: String?
        let created: Double?
        let updated: Double?
        let preview: String?
        let turns: Int?
        let processing: Bool?

        var id: String { sessionID }

        enum CodingKeys: String, CodingKey {
            case sessionID = "session_id"
            case tag, created, updated, preview, turns, processing
        }

        /// What the list shows. A session that was opened and never used has
        /// no name and no preview; the server keeps it, so the app names it
        /// rather than showing a blank row.
        var displayTitle: String {
            if let tag, !tag.isEmpty { return tag }
            if let preview, !preview.isEmpty { return preview }
            return "New session"
        }

        var updatedDate: Date {
            Date(timeIntervalSince1970: updated ?? created ?? 0)
        }
    }

    struct HistoryMessage: Sendable, Decodable, Identifiable, Equatable {
        let role: String
        let content: String
        var id: String { role + "\u{0}" + content.prefix(64) }
    }

    struct History: Sendable, Decodable {
        let sessionID: String
        let processing: Bool
        let messages: [HistoryMessage]

        enum CodingKeys: String, CodingKey {
            case sessionID = "session_id"
            case processing, messages
        }
    }

    struct Config: Sendable, Decodable {
        let title: String?
        let brand: String?
        let agent: String?
        let authEnabled: Bool?
        let ratingEnabled: Bool?
        let showLogs: Bool?

        enum CodingKeys: String, CodingKey {
            case title, brand, agent
            case authEnabled = "auth_enabled"
            case ratingEnabled = "rating_enabled"
            case showLogs = "show_logs"
        }
    }

    // MARK: - Requests

    func config() async throws -> Config {
        try await get("api/config")
    }

    func sessions() async throws -> [SessionSummary] {
        struct Envelope: Decodable { let sessions: [SessionSummary] }
        return try await get("api/sessions", as: Envelope.self).sessions
    }

    func newSession() async throws -> String {
        struct Envelope: Decodable { let sessionID: String
            enum CodingKeys: String, CodingKey { case sessionID = "session_id" }
        }
        return try await post("api/sessions/new", body: [:], as: Envelope.self).sessionID
    }

    func history(session: String) async throws -> History {
        try await get("api/history", session: session)
    }

    func rename(session: String, tag: String) async throws {
        _ = try await send(
            "api/sessions/\(session)", method: "PATCH",
            body: ["tag": tag], session: session
        )
    }

    func delete(session: String) async throws {
        _ = try await send("api/sessions/\(session)", method: "DELETE", session: session)
    }

    func stop(session: String) async throws {
        _ = try await send("api/chat/stop", method: "POST", body: [:], session: session)
    }

    func rate(session: String, turn: Int, rating: Int?) async throws {
        var body: [String: Any] = ["turn": turn]
        body["rating"] = rating ?? NSNull()
        _ = try await send("api/rating", method: "POST", body: body, session: session)
    }

    /// Answer a pending approval.
    ///
    /// `decision` is one of `once`, `session`, `deny`. The server treats
    /// anything else — and anything that never arrives — as a deny, so a
    /// failure here is already the safe outcome.
    func approve(session: String, approvalID: String, decision: String) async throws {
        _ = try await send(
            "api/approval", method: "POST",
            body: ["approval_id": approvalID, "decision": decision],
            session: session
        )
    }

    /// Upload one file into the session's data directory.
    ///
    /// Returns the absolute path the server stored it at, which is what
    /// `POST /api/chat` expects in its `files` array.
    func upload(session: String, fileURL: URL) async throws -> String {
        let boundary = "onit-" + UUID().uuidString
        var body = Data()
        let filename = fileURL.lastPathComponent
        let fileData = try Data(contentsOf: fileURL)

        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append(
            "Content-Disposition: form-data; name=\"file\"; filename=\"\(filename)\"\r\n"
                .data(using: .utf8)!
        )
        body.append("Content-Type: application/octet-stream\r\n\r\n".data(using: .utf8)!)
        body.append(fileData)
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)

        var request = URLRequest(url: baseURL.appending(path: "api/upload"))
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.setValue(session, forHTTPHeaderField: "X-Session-Id")
        request.httpBody = body

        let (data, response) = try await URLSession.shared.data(for: request)
        try Self.check(response, data: data)
        struct Envelope: Decodable { let path: String? }
        guard let path = (try? JSONDecoder().decode(Envelope.self, from: data))?.path else {
            throw OnItError.server("Upload did not return a path")
        }
        return path
    }

    /// Stream one turn.
    ///
    /// Returns an `AsyncThrowingStream` of events. Cancelling the consuming
    /// task cancels the request, which is what Stop does.
    func chat(session: String, message: String, files: [String]) -> AsyncThrowingStream<ChatEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var request = URLRequest(url: baseURL.appending(path: "api/chat"))
                    request.httpMethod = "POST"
                    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                    request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
                    request.setValue(session, forHTTPHeaderField: "X-Session-Id")
                    request.httpBody = try JSONSerialization.data(
                        withJSONObject: ["message": message, "files": files]
                    )
                    // A long agent run can sit silent for minutes between tool
                    // calls; the server's keepalive is what keeps this alive,
                    // so the client timeout only has to outlast the gap.
                    request.timeoutInterval = 600

                    let (bytes, response) = try await URLSession.shared.bytes(for: request)
                    try Self.check(response, data: nil)

                    let reader = SSEReader()
                    var chunk = Data()
                    chunk.reserveCapacity(4096)

                    for try await byte in bytes {
                        chunk.append(byte)
                        // Flush on a frame boundary rather than per byte: the
                        // parser is an actor and a hop per byte would dominate
                        // the cost of a fast stream.
                        if byte == 0x0A, chunk.count >= 2,
                           chunk[chunk.count - 2] == 0x0A {
                            for event in await reader.consume(chunk) {
                                continuation.yield(event)
                            }
                            chunk.removeAll(keepingCapacity: true)
                        } else if chunk.count > 8192 {
                            for event in await reader.consume(chunk) {
                                continuation.yield(event)
                            }
                            chunk.removeAll(keepingCapacity: true)
                        }
                    }
                    if !chunk.isEmpty {
                        for event in await reader.consume(chunk) {
                            continuation.yield(event)
                        }
                    }
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    // MARK: - Plumbing

    private func get<T: Decodable>(_ path: String, session: String? = nil, as type: T.Type = T.self) async throws -> T {
        let data = try await send(path, method: "GET", session: session)
        return try JSONDecoder().decode(T.self, from: data)
    }

    private func post<T: Decodable>(_ path: String, body: [String: Any], as type: T.Type) async throws -> T {
        let data = try await send(path, method: "POST", body: body)
        return try JSONDecoder().decode(T.self, from: data)
    }

    @discardableResult
    private func send(
        _ path: String, method: String,
        body: [String: Any]? = nil, session: String? = nil
    ) async throws -> Data {
        var request = URLRequest(url: baseURL.appending(path: path))
        request.httpMethod = method
        if let session { request.setValue(session, forHTTPHeaderField: "X-Session-Id") }
        if let body {
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
        }
        let (data, response) = try await URLSession.shared.data(for: request)
        try Self.check(response, data: data)
        return data
    }

    /// Turn a non-2xx response into an error carrying the server's own
    /// `detail` string, so the UI shows the backend's words rather than a
    /// status code the user cannot act on.
    static func check(_ response: URLResponse, data: Data?) throws {
        guard let http = response as? HTTPURLResponse else { return }
        guard (200..<300).contains(http.statusCode) else {
            var detail = HTTPURLResponse.localizedString(forStatusCode: http.statusCode)
            if let data, let json = JSON.object(from: String(decoding: data, as: UTF8.self)),
               let serverDetail = json["detail"] as? String {
                detail = serverDetail
            }
            throw OnItError.server(detail, status: http.statusCode)
        }
    }
}

enum OnItError: LocalizedError {
    case server(String, status: Int = 0)
    case notReady

    var errorDescription: String? {
        switch self {
        case let .server(message, status):
            return status == 409 ? "A task is already running in this session." : message
        case .notReady:
            return "The OnIt backend is not ready yet."
        }
    }
}
