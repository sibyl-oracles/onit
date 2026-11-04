import Foundation

/// Turns a `URLSession` byte stream into `ChatEvent`s.
///
/// An `actor` because the parser holds mutable state (the partial line buffer)
/// that the delegate feeds from a background queue and the UI reads from the
/// main actor. Serialising it here is cheaper than a lock and matches Swift 6
/// strict concurrency without a single `@unchecked`.
///
/// No third-party SSE package: the format is three lines of parsing and the
/// supply-chain cost of a dependency is not worth it.
actor SSEReader {
    private var buffer = ""
    private var eventName = ""
    private var dataLines: [String] = []

    /// Feed raw bytes; returns whatever events became complete.
    ///
    /// A frame is terminated by a blank line. `data:` lines within one frame
    /// are joined with newlines, per the SSE spec, which matters because a
    /// `done` payload containing a multi-line answer is still one `data:` line
    /// from this server but need not stay that way.
    func consume(_ bytes: Data) -> [ChatEvent] {
        guard let chunk = String(data: bytes, encoding: .utf8) else { return [] }
        buffer += chunk
        var events: [ChatEvent] = []

        while let newline = buffer.firstIndex(of: "\n") {
            var line = String(buffer[buffer.startIndex..<newline])
            buffer = String(buffer[buffer.index(after: newline)...])
            if line.hasSuffix("\r") { line.removeLast() }

            if line.isEmpty {
                if let event = flush() { events.append(event) }
                continue
            }
            // A comment line: the server sends ": keepalive" every 15 s to hold
            // the connection open through a proxy. It carries no event.
            if line.hasPrefix(":") { continue }

            if let value = field("event", in: line) {
                eventName = value
            } else if let value = field("data", in: line) {
                dataLines.append(value)
            }
            // `id:` and `retry:` are not sent by this server and are ignored.
        }
        return events
    }

    private func field(_ name: String, in line: String) -> String? {
        let prefix = name + ":"
        guard line.hasPrefix(prefix) else { return nil }
        var value = String(line.dropFirst(prefix.count))
        if value.hasPrefix(" ") { value.removeFirst() }
        return value
    }

    private func flush() -> ChatEvent? {
        defer {
            eventName = ""
            dataLines.removeAll(keepingCapacity: true)
        }
        guard !dataLines.isEmpty else { return nil }
        let name = eventName.isEmpty ? "message" : eventName
        return ChatEvent.decode(name: name, data: dataLines.joined(separator: "\n"))
    }
}
