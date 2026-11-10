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
        /// Attachment descriptors for files a turn produced. The server only
        /// sends them for assistant rows; nil elsewhere, and nil from an
        /// older sidecar that does not send the key at all.
        let files: [FileInfo]?

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
        let ttsEnabled: Bool?

        enum CodingKeys: String, CodingKey {
            case title, brand, agent
            case authEnabled = "auth_enabled"
            case ratingEnabled = "rating_enabled"
            case showLogs = "show_logs"
            case ttsEnabled = "tts_enabled"
        }
    }

    // MARK: - Endpoints and keys

    /// Where one secret lives, as the server reports it.
    ///
    /// The value itself never crosses the wire — `masked` is at most the last
    /// four characters, the same `••••abcd` the terminal prints. `source`
    /// tells the user where it comes from: the keychain, an environment
    /// variable, a legacy secret name, or the config file itself.
    struct KeyStatus: Sendable, Decodable, Equatable {
        let source: String?
        let masked: String?
        let label: String

        var isSet: Bool { source != nil }
    }

    /// One configured model endpoint, in preference order.
    struct EndpointRow: Sendable, Decodable, Identifiable, Equatable {
        let id: String
        let host: String
        let model: String?
        let name: String?
        let priority: Int
        let position: Int
        let key: KeyStatus
        let notes: [String]
    }

    /// A one-click host in the add form.
    struct EndpointPreset: Sendable, Decodable, Identifiable, Equatable {
        let label: String
        let host: String
        let needsKey: Bool

        var id: String { host }

        enum CodingKeys: String, CodingKey {
            case label, host
            case needsKey = "needs_key"
        }
    }

    /// One integration secret (Tavily, Ollama, GitHub, …).
    ///
    /// The server's `label` field here is the *status* text, not a display
    /// name — the status dict is spread over the row and wins. The display
    /// name is derived from the key, which is the stable identifier.
    struct IntegrationRow: Sendable, Decodable, Identifiable, Equatable {
        let key: String
        let label: String
        let envVar: String?
        let blurb: String
        let source: String?
        let masked: String?

        var id: String { key }

        enum CodingKeys: String, CodingKey {
            case key, label, blurb, source, masked
            case envVar = "env_var"
        }

        /// The status the server means: `label` is the status text.
        var status: String { label }
        var isSet: Bool { source != nil }

        /// The human name for the row, derived from the key spelling.
        var displayName: String {
            let stem = key
                .replacingOccurrences(of: "_api_key", with: "")
                .replacingOccurrences(of: "_token", with: "")
            return stem
                .split(separator: "_")
                .map { $0.capitalized }
                .joined(separator: " ")
        }
    }

    /// Everything the Keys & Models tab renders, from `GET /api/endpoints`.
    struct EndpointSurface: Sendable, Decodable, Equatable {
        let endpoints: [EndpointRow]
        let presets: [EndpointPreset]
        let integrations: [IntegrationRow]
        let configPath: String?
        let keychain: Bool

        enum CodingKeys: String, CodingKey {
            case endpoints, presets, integrations, keychain
            case configPath = "config_path"
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

    /// The bytes behind an attachment (`/uploads/...`), for inline playback.
    ///
    /// `send` already returns raw data; this is the public, JSON-free name
    /// for it, because an attachment is not a decoded payload.
    func data(_ path: String) async throws -> Data {
        try await send(path, method: "GET")
    }

    // MARK: - Endpoints and keys

    /// Everything the Keys & Models tab shows.
    func endpointSurface() async throws -> EndpointSurface {
        try await get("api/endpoints")
    }

    /// Replace the endpoint list, in the order they should be tried.
    ///
    /// Keys are not part of this call — they are addressed by URL and stored
    /// separately, so reordering or renaming cannot lose one.
    @discardableResult
    func saveEndpoints(_ rows: [EndpointRow]) async throws -> EndpointSurface {
        let body: [String: Any] = [
            "endpoints": rows.map { row -> [String: Any] in
                var entry: [String: Any] = ["host": row.host]
                if let model = row.model, !model.isEmpty { entry["model"] = model }
                if let name = row.name, !name.isEmpty { entry["name"] = name }
                entry["priority"] = row.priority
                return entry
            }
        ]
        return try await put("api/endpoints", body: body, as: EndpointSurface.self)
    }

    /// Store one endpoint's API key. The key crosses the wire once, on save,
    /// and lands in the OS keychain on the far side; the app never keeps it.
    @discardableResult
    func storeEndpointKey(host: String, key: String) async throws -> KeyStatus {
        try await put(
            "api/endpoints/key",
            body: ["host": host, "key": key],
            as: KeyStatus.self
        )
    }

    @discardableResult
    func forgetEndpointKey(host: String) async throws -> KeyStatus {
        try await delete("api/endpoints/key", query: ["host": host], as: KeyStatus.self)
    }

    @discardableResult
    func storeIntegrationKey(keyName: String, value: String) async throws -> KeyStatus {
        try await put(
            "api/integrations/key",
            body: ["key_name": keyName, "key": value],
            as: KeyStatus.self
        )
    }

    @discardableResult
    func forgetIntegrationKey(keyName: String) async throws -> KeyStatus {
        try await delete("api/integrations/key", query: ["key_name": keyName], as: KeyStatus.self)
    }

    // MARK: - Text to speech

    /// Speak *text* through the server's local Kokoro model.
    ///
    /// Returns the raw WAV bytes, played by `TtsCenter`. The server renders
    /// the whole clip before answering — synthesis is not streamed — so the
    /// caller decides what a wait is worth: this is called after a reply has
    /// landed, never while the user is still reading it.
    func tts(_ text: String) async throws -> Data {
        try await send("api/tts", method: "POST", body: ["text": text])
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

                    // Byte-at-a-time into the reader, not `bytes.lines`:
                    // AsyncLineSequence drops empty lines, and the empty line
                    // is the SSE frame boundary — through `.lines` no frame
                    // ever completes, so no event ever fires and the answer
                    // only shows up after a restart via /api/history. The
                    // byte path buffers, splits on raw UTF-8 (CRLF-safe) and
                    // flushes on the boundary; it is the path the tests drive.
                    let reader = SSEReader()
                    var pending: [UInt8] = []
                    pending.reserveCapacity(4096)
                    for try await byte in bytes {
                        pending.append(byte)
                        if byte == 0x0A {  // newline: flush whole lines at once
                            for event in await reader.consume(Data(pending)) {
                                continuation.yield(event)
                            }
                            pending.removeAll(keepingCapacity: true)
                        }
                    }
                    if !pending.isEmpty {
                        for event in await reader.consume(Data(pending)) {
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

    /// A cheap liveness probe with a short timeout, for the health monitor.
    ///
    /// `config()` shares the session's default 60 s timeout, which is right
    /// for real requests and wrong for a poll: an offline host would hold
    /// each health check for a minute.
    func ping() async -> Bool {
        var request = URLRequest(url: baseURL.appending(path: "api/config"))
        request.timeoutInterval = 3
        request.cachePolicy = .reloadIgnoringLocalCacheData
        guard let (_, response) = try? await URLSession.shared.data(for: request),
              let http = response as? HTTPURLResponse
        else { return false }
        return (200..<300).contains(http.statusCode)
    }

    private func get<T: Decodable>(_ path: String, session: String? = nil, as type: T.Type = T.self) async throws -> T {
        let data = try await send(path, method: "GET", session: session)
        return try JSONDecoder().decode(T.self, from: data)
    }

    private func post<T: Decodable>(_ path: String, body: [String: Any], as type: T.Type) async throws -> T {
        let data = try await send(path, method: "POST", body: body)
        return try JSONDecoder().decode(T.self, from: data)
    }

    private func put<T: Decodable>(_ path: String, body: [String: Any], as type: T.Type) async throws -> T {
        let data = try await send(path, method: "PUT", body: body)
        return try JSONDecoder().decode(T.self, from: data)
    }

    private func delete<T: Decodable>(_ path: String, query: [String: String], as type: T.Type) async throws -> T {
        var components = URLComponents(url: baseURL.appending(path: path), resolvingAgainstBaseURL: false)!
        if !query.isEmpty {
            components.queryItems = query.map { URLQueryItem(name: $0.key, value: $0.value) }
        }
        var request = URLRequest(url: components.url!)
        request.httpMethod = "DELETE"
        let (data, response) = try await URLSession.shared.data(for: request)
        try Self.check(response, data: data)
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
