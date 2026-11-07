import XCTest
@testable import OnIt

/// The SSE parser, fed the ways real streams arrive: whole frames, frames
/// sliced mid-line, CRLF endings, keepalive comments, multi-line data.
///
/// The byte path (`consume(_:)`) is what a hand-rolled transport would use;
/// the line path (`consume(line:)`) is what `URLSession.bytes.lines` feeds.
/// Both must produce identical events for identical input.
final class SSEReaderTests: XCTestCase {
    // MARK: - Byte path

    func testWholeFrameProducesOneEvent() async throws {
        let reader = SSEReader()
        let events = await reader.consume(Data("event: token\ndata: {\"delta\": \"Hel\"}\n\n".utf8))
        XCTAssertEqual(events, [.token("Hel")])
    }

    func testByteSlicedAcrossChunkBoundaries() async throws {
        let reader = SSEReader()
        let frame = "event: token\ndata: {\"delta\": \"Hello\"}\n\n"
        var events: [ChatEvent] = []
        // One byte at a time is the worst case a transport can produce.
        for byte in frame.utf8 {
            events.append(contentsOf: await reader.consume(Data([byte])))
        }
        XCTAssertEqual(events, [.token("Hello")])
    }

    func testCRLFLineEndings() async throws {
        let reader = SSEReader()
        let events = await reader.consume(Data("event: token\r\ndata: {\"delta\": \"Hi\"}\r\n\r\n".utf8))
        XCTAssertEqual(events, [.token("Hi")])
    }

    func testCommentLinesCarryNoEvent() async throws {
        let reader = SSEReader()
        var events = await reader.consume(Data(": keepalive\n\n".utf8))
        XCTAssertTrue(events.isEmpty)
        events = await reader.consume(Data("event: token\ndata: {\"delta\": \"a\"}\n: keepalive\n\n".utf8))
        XCTAssertEqual(events, [.token("a")])
    }

    func testMultiLineDataJoinedWithNewlines() async throws {
        let reader = SSEReader()
        var events = await reader.consume(Data("event: token\ndata: {\"delta\": \"a\"\ndata: {\"delta\": \"b\"}\n\n".utf8))
        // Two `data:` lines in one frame are joined with a newline, per the
        // SSE spec — this server never sends them (json.dumps is one line),
        // but the parser must not lose the first when a second arrives.
        guard case let .unknown(name, data) = events.first else {
            return XCTFail("expected .unknown, got \(String(describing: events.first))")
        }
        XCTAssertEqual(name, "token")
        XCTAssertEqual(data, "{\"delta\": \"a\"\n{\"delta\": \"b\"}")
    }

    func testBareContinuationLineIsIgnored() async throws {
        // A line that is not `event:`/`data:`/a comment is not part of the
        // payload. The frame still flushes with what it has.
        let reader = SSEReader()
        var events = await reader.consume(Data("event: token\ndata: {\"delta\": \"a\"\n \"x\"}\n\n".utf8))
        guard case let .unknown(_, data) = events.first else {
            return XCTFail("expected an event, got \(String(describing: events.first))")
        }
        XCTAssertEqual(data, "{\"delta\": \"a\"")
    }

    func testPartialLineHeldUntilTerminator() async throws {
        let reader = SSEReader()
        var events = await reader.consume(Data("event: token\ndata: {\"delta\": \"end\"}".utf8))
        XCTAssertTrue(events.isEmpty, "a line without its newline is not a line yet")
        events = await reader.consume(Data("\n\n".utf8))
        XCTAssertEqual(events, [.token("end")])
    }

    func testConsecutiveFrames() async throws {
        let reader = SSEReader()
        var events = await reader.consume(
            Data("event: token\ndata: {\"delta\": \"a\"}\n\nevent: token\ndata: {\"delta\": \"b\"}\n\n".utf8)
        )
        XCTAssertEqual(events, [.token("a"), .token("b")])
        events = await reader.consume(Data("event: token\ndata: {\"delta\": \"c\"}\n\n".utf8))
        XCTAssertEqual(events, [.token("c")], "state must reset between frames")
    }

    // MARK: - Line path

    func testLinePathMatchesBytePath() async throws {
        let frame = "event: token\ndata: {\"delta\": \"Hello\"}\n\n"
        let byteReader = SSEReader()
        let lineReader = SSEReader()
        let fromBytes = await byteReader.consume(Data(frame.utf8))
        var fromLines: [ChatEvent] = []
        for line in frame.components(separatedBy: "\n") {
            fromLines.append(contentsOf: await lineReader.consume(line: line))
        }
        XCTAssertEqual(fromBytes, fromLines)
        XCTAssertEqual(fromLines, [.token("Hello")])
    }

    func testLinePathStripsCR() async throws {
        let reader = SSEReader()
        var events = await reader.consume(line: "event: token\r")
        XCTAssertTrue(events.isEmpty)
        events = await reader.consume(line: "data: {\"delta\": \"x\"}\r")
        XCTAssertTrue(events.isEmpty)
        events = await reader.consume(line: "\r")
        XCTAssertEqual(events, [.token("x")])
    }

    func testEventWithoutDataIsDropped() async throws {
        // The server always pairs an event line with a data line, but a frame
        // without data must not crash the reader or leak into the next frame.
        let reader = SSEReader()
        var events = await reader.consume(line: "event: answer_start")
        XCTAssertTrue(events.isEmpty)
        events = await reader.consume(line: "data: {}")
        XCTAssertTrue(events.isEmpty)
        events = await reader.consume(line: "")
        XCTAssertEqual(events, [.answerStart])
    }
}
