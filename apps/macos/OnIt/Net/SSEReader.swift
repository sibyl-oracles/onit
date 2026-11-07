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
    /// Raw bytes not yet split into lines. Held as plain bytes and scanned
    /// manually — see `consume(_:)` for why this must not be a `String`, and
    /// `Data`'s index semantics are their own trap after mutation.
    private var buffer: [UInt8] = []
    private var eventName = ""
    private var dataLines: [String] = []

    /// Feed raw bytes; returns whatever events became complete.
    ///
    /// Kept as the byte-level entry point — the tests drive it with
    /// deliberately awkward slices — but the live stream goes through
    /// `consume(line:)`, one complete line at a time.
    ///
    /// Splitting happens on raw UTF-8, not on `String` indices: in Swift a
    /// `\r\n` pair is one grapheme cluster, so `firstIndex(of: "\n")` never
    /// matches a CRLF stream and a proxy that normalized line endings would
    /// silently deliver nothing. Bytes do not have that problem.
    func consume(_ bytes: Data) -> [ChatEvent] {
        buffer.append(contentsOf: bytes)

        var events: [ChatEvent] = []
        var lineStart = 0
        while let newline = buffer[lineStart...].firstIndex(of: 0x0A) {
            let line = String(decoding: buffer[lineStart..<newline], as: UTF8.self)
            lineStart = newline + 1
            events.append(contentsOf: consume(line: line))
        }
        // Keep only the unconsumed tail so the buffer cannot grow unbounded.
        if lineStart > 0 {
            buffer.removeFirst(lineStart)
        }
        return events
    }

    /// Feed one complete line (no terminator); returns whatever events it
    /// completed. An empty line is the frame boundary.
    func consume(line: String) -> [ChatEvent] {
        var line = line
        if line.hasSuffix("\r") { line.removeLast() }

        if line.isEmpty {
            if let event = flush() { return [event] }
            return []
        }
        // A comment line: the server sends ": keepalive" every 15 s to hold
        // the connection open through a proxy. It carries no event.
        if line.hasPrefix(":") { return [] }

        if let value = field("event", in: line) {
            eventName = value
        } else if let value = field("data", in: line) {
            dataLines.append(value)
        }
        // `id:` and `retry:` are not sent by this server and are ignored.
        return []
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
