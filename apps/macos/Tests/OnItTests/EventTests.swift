import XCTest
@testable import OnIt

/// Event decoding against the schema `onit/ui/api.py` documents — the same
/// fixtures the server actually emits, including the empty payload on
/// `answer_start`.
final class EventTests: XCTestCase {
    private func decode(_ name: String, _ data: String) -> ChatEvent {
        ChatEvent.decode(name: name, data: data)
    }

    func testTokenAndThink() {
        XCTAssertEqual(decode("token", #"{"delta": "Hi"}"#), .token("Hi"))
        XCTAssertEqual(decode("think", #"{"delta": "hmm"}"#), .think("hmm"))
    }

    func testStatusAndPhaseEnd() {
        XCTAssertEqual(decode("status", #"{"text": "Running shell"}"#), .status("Running shell"))
        XCTAssertEqual(decode("phase_end", #"{"content": "done"}"#), .phaseEnd("done"))
    }

    func testAnswerStartWithEmptyPayload() {
        // The server emits `answer_start` with `{}` — the reader must not
        // drop it for having no fields.
        XCTAssertEqual(decode("answer_start", "{}"), .answerStart)
    }

    func testApprovalWithIntegerExpiresIn() {
        // Python's json.dumps writes the default timeout as an int; the
        // decoder must accept both int and double.
        let event = decode(
            "approval",
            #"{"approval_id": "ap-1", "command": "rm -rf build", "reason": "clean", "subjects": ["build"], "expires_in": 30}"#
        )
        guard case let .approval(request) = event else {
            return XCTFail("expected .approval, got \(event)")
        }
        XCTAssertEqual(request.id, "ap-1")
        XCTAssertEqual(request.command, "rm -rf build")
        XCTAssertEqual(request.subjects, ["build"])
        XCTAssertEqual(request.expiresIn, 30)
    }

    func testApprovalClosedAndError() {
        XCTAssertEqual(decode("approval_closed", #"{"approval_id": "ap-1"}"#), .approvalClosed("ap-1"))
        XCTAssertEqual(decode("error", #"{"message": "boom"}"#), .error("boom"))
    }

    func testDonePayload() {
        let event = decode(
            "done",
            #"{"content": "Answer", "files": [{"name": "a.txt", "url": "/f/a.txt", "size": 3, "kind": "file"}], "elapsed": 1.5, "tok_s": 40.0, "turn": 2, "rating_enabled": true, "timing": {"turns": 1, "tool_calls": 3, "model_s": 1.2, "prompt_tokens_total": 90, "completion_tokens": 45, "token_stats": "135 tok"}}"#
        )
        guard case let .done(payload) = event else {
            return XCTFail("expected .done, got \(event)")
        }
        XCTAssertEqual(payload.content, "Answer")
        XCTAssertEqual(payload.files.count, 1)
        XCTAssertEqual(payload.files.first?.name, "a.txt")
        XCTAssertEqual(payload.elapsed, 1.5)
        XCTAssertEqual(payload.tokensPerSecond, 40.0)
        XCTAssertEqual(payload.turn, 2)
        XCTAssertTrue(payload.ratingEnabled)
        XCTAssertEqual(payload.timing.toolCalls, 3)
        XCTAssertEqual(payload.timing.tokenStats, "135 tok")
    }

    func testDoneWithMissingFieldsDefaults() {
        guard case let .done(payload) = decode("done", "{}") else {
            return XCTFail("expected .done")
        }
        XCTAssertEqual(payload.content, "")
        XCTAssertTrue(payload.files.isEmpty)
        XCTAssertEqual(payload.turn, 0)
        XCTAssertFalse(payload.ratingEnabled)
    }

    func testUnknownEventNameIsKept() {
        guard case let .unknown(name, data) = decode("new_thing", #"{"x": 1}"#) else {
            return XCTFail("expected .unknown")
        }
        XCTAssertEqual(name, "new_thing")
        XCTAssertEqual(data, #"{"x": 1}"#)
    }

    func testNonJSONDataSurvivesAsUnknown() {
        // A torn or non-JSON frame must be visible, not silently dropped.
        guard case let .unknown(name, data) = decode("token", "not json") else {
            return XCTFail("expected .unknown")
        }
        XCTAssertEqual(name, "token")
        XCTAssertEqual(data, "not json")
    }
}

final class AudioFileInfoTests: XCTestCase {
    func testDonePayloadDecodesAudioKindAndMime() {
        let json = """
        {"content": "ok", "files": [{"name": "take.wav", "url": "/uploads/s/take.wav",
          "size": 88, "kind": "audio", "mime": "audio/wav"}], "elapsed": 1, "tok_s": 0,
         "turn": 1, "rating_enabled": false, "timing": {}}
        """
        guard let json2 = JSON.object(from: json) else { return XCTFail("json") }
        let payload = DonePayload(json2)
        let file = payload.files.first
        XCTAssertEqual(file?.kind, "audio")
        XCTAssertEqual(file?.mime, "audio/wav")
        XCTAssertEqual(file?.isAudio, true)
        XCTAssertEqual(file?.isImage, false)
    }

    func testDonePayloadDecodesImageKind() {
        let json = """
        {"content": "ok", "files": [{"name": "chart.png", "url": "/uploads/s/chart.png",
          "size": 2048, "kind": "image", "preview": "a chart"}], "elapsed": 1, "tok_s": 0,
         "turn": 1, "rating_enabled": false, "timing": {}}
        """
        guard let json2 = JSON.object(from: json) else { return XCTFail("json") }
        let payload = DonePayload(json2)
        let file = payload.files.first
        XCTAssertEqual(file?.kind, "image")
        XCTAssertEqual(file?.isImage, true)
        XCTAssertEqual(file?.isAudio, false)
        XCTAssertEqual(file?.preview, "a chart")
    }

    func testHistoryMessageDecodesFilesAndToleratesTheirAbsence() throws {
        let withFiles = """
        {"role": "assistant", "content": "here", "files":
          [{"name": "song.mp3", "url": "/uploads/s/song.mp3", "size": 9, "kind": "audio", "mime": "audio/mpeg"}]}
        """
        let decoded = try JSONDecoder().decode(OnItClient.HistoryMessage.self, from: withFiles.data(using: .utf8)!)
        XCTAssertEqual(decoded.files?.first?.mime, "audio/mpeg")

        let withoutFiles = #"{"role": "user", "content": "hi"}"#
        let bare = try JSONDecoder().decode(OnItClient.HistoryMessage.self, from: withoutFiles.data(using: .utf8)!)
        XCTAssertNil(bare.files)
    }
}

