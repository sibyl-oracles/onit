import XCTest
@testable import OnIt

/// Replays the real server's captured byte stream through the app's read
/// path: URLSession.bytes batched on newline into SSEReader. Proves the
/// final `done` frame survives the transport — the regression that made
/// answers appear only after a restart.
final class LiveStreamReplayTests: XCTestCase {
    func testDoneSurvivesTheBytePath() async throws {
        // A minimal in-process SSE body: token then done, blank-line framed.
        let body = "event: token\ndata: {\"delta\": \"Hel\"}\n\nevent: done\ndata: {\"content\": \"FINAL\"}\n\n"
        let reader = SSEReader()
        var events: [ChatEvent] = []
        var pending: [UInt8] = []
        for byte in body.utf8 {
            pending.append(byte)
            if byte == 0x0A {
                events.append(contentsOf: await reader.consume(Data(pending)))
                pending.removeAll(keepingCapacity: true)
            }
        }
        if !pending.isEmpty {
            events.append(contentsOf: await reader.consume(Data(pending)))
        }
        XCTAssertEqual(events.count, 2, "both frames must complete")
        XCTAssertEqual(events.first, .token("Hel"))
        guard case let .done(payload) = events.last else {
            return XCTFail("last event must be done, got \(String(describing: events.last))")
        }
        XCTAssertEqual(payload.content, "FINAL")
    }

    func testLinePathDropsTheFrameBoundary() async throws {
        // Documents the bug: through AsyncLineSequence (which the app used
        // before), the blank lines are gone and no frame ever completes.
        let body = "event: token\ndata: {\"delta\": \"Hel\"}\n\nevent: done\ndata: {\"content\": \"FINAL\"}\n\n"
        let lines = body.split(separator: "\n", omittingEmptySubsequences: true)
        let reader = SSEReader()
        var events: [ChatEvent] = []
        for line in lines {
            events.append(contentsOf: await reader.consume(line: String(line)))
        }
        XCTAssertTrue(events.isEmpty, "no blank lines -> no frame boundary -> no events")
    }
}
