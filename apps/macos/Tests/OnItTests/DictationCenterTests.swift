import XCTest
@testable import OnIt

@MainActor
final class DictationCenterTests: XCTestCase {
    /// The accumulator: a finalized utterance becomes the base, and the
    /// in-flight hypothesis rides alongside it. One breath between sentences
    /// finalizes the first — the join must still read as one dictation.
    func testFinalAndPartialJoin() {
        let center = DictationCenter()
        XCTAssertEqual(center.currentText, "")

        center.updatePartial("what should")
        XCTAssertEqual(center.currentText, "what should")

        center.commitFinal("what should")
        center.updatePartial("onit do")
        XCTAssertEqual(center.currentText, "what should onit do")

        center.commitFinal("what should onit do")
        XCTAssertEqual(center.currentText, "what should onit do")
    }

    /// A final result replaces any partial it supersedes: the hypothesis is
    /// a preview of the same words, never an addition to them.
    func testFinalReplacesPartial() {
        let center = DictationCenter()
        center.updatePartial("hello wor")
        center.commitFinal("hello world")
        XCTAssertEqual(center.currentText, "hello world")
        XCTAssertFalse(center.currentText.contains("wor hello"))
    }

    /// Reset clears everything: after a send, a stale transcript must not
    /// resurface in the next dictation session.
    func testResetClearsAccumulator() {
        let center = DictationCenter()
        center.commitFinal("first pass")
        center.updatePartial("se")
        center.reset()
        XCTAssertEqual(center.currentText, "")
        XCTAssertNil(center.snapshot.error)
    }

    /// Stopping while not listening is a no-op, not a state corruption.
    func testStopWhenIdleIsNoOp() {
        let center = DictationCenter()
        center.stop()
        XCTAssertFalse(center.snapshot.isListening)
        XCTAssertEqual(center.currentText, "")
    }

    /// The snapshot is the composer's whole view of the center: listening,
    /// live text, error and denial travel together so the view orders its
    /// reactions against one value.
    func testSnapshotCarriesState() {
        let center = DictationCenter()
        var snap = center.snapshot
        XCTAssertFalse(snap.isListening)
        XCTAssertFalse(snap.isDenied)
        XCTAssertNil(snap.error)
        XCTAssertEqual(snap.currentText, "")

        center.commitFinal("note to self")
        snap = center.snapshot
        XCTAssertEqual(snap.currentText, "note to self")
    }
}

@MainActor
final class VoiceCommandTests: XCTestCase {
    // MARK: - "run it" / "execute"

    /// The plain execute command, with and without punctuation.
    func testRunIt() {
        XCTAssertEqual(VoiceCommand.consume(from: "run it")?.command, .run)
        XCTAssertEqual(VoiceCommand.consume(from: "Run it.")?.command, .run)
        XCTAssertEqual(VoiceCommand.consume(from: "execute")?.command, .run)
        XCTAssertEqual(VoiceCommand.consume(from: "Execute!")?.command, .run)
    }

    /// The command peels off the tail, leaving the task text behind.
    func testRunItLeavesTaskText() {
        let match = VoiceCommand.consume(from: "summarize this chapter, run it")
        XCTAssertEqual(match?.command, .run)
        XCTAssertEqual(match?.remaining, "summarize this chapter")
    }

    /// Filler words must not block the match.
    func testRunItWithFillers() {
        XCTAssertEqual(VoiceCommand.consume(from: "um, run it")?.command, .run)
        XCTAssertEqual(VoiceCommand.consume(from: "hey execute please")?.command, .run)
    }

    // MARK: - "onit, stop" barge-in

    /// Both spellings of the anchor: the recognizer is inconsistent about
    /// whether "onit" comes back as one word or the English phrase.
    func testOnitStop() {
        XCTAssertEqual(VoiceCommand.consume(from: "onit stop")?.command, .stop)
        XCTAssertEqual(VoiceCommand.consume(from: "on it stop")?.command, .stop)
        XCTAssertEqual(VoiceCommand.consume(from: "Onit, stop.")?.command, .stop)
    }

    /// "stop it" is an accepted barge-in variant.
    func testStopIt() {
        XCTAssertEqual(VoiceCommand.consume(from: "stop it")?.command, .stop)
    }

    /// Bare "stop" is deliberately not a command: it appears in half of all
    /// task text, and a false stop kills real work.
    func testBareStopIsNotACommand() {
        XCTAssertNil(VoiceCommand.consume(from: "stop"))
        XCTAssertNil(VoiceCommand.consume(from: "stop the server"))
    }

    /// A task that merely mentions the command words mid-sentence must not
    /// fire — only the tail counts.
    func testMidSentenceMentionDoesNotFire() {
        XCTAssertNil(VoiceCommand.consume(from: "run it and then stop the server"))
        XCTAssertNil(VoiceCommand.consume(from: "how do I run it in production"))
        XCTAssertNil(VoiceCommand.consume(from: "tell onit to stop the build"))
    }

    /// Ordinary text produces no command and no rewrite.
    func testOrdinaryTextUnmatched() {
        XCTAssertNil(VoiceCommand.consume(from: "write me a haiku about the sea"))
        XCTAssertNil(VoiceCommand.consume(from: ""))
        XCTAssertNil(VoiceCommand.consume(from: "   "))
    }

    /// The stop command strips cleanly, though a barge-in normally sends
    /// nothing afterwards — the remaining text is what stays in the draft.
    func testStopLeavesRemainingText() {
        let match = VoiceCommand.consume(from: "wait, on it stop")
        XCTAssertEqual(match?.command, .stop)
        XCTAssertEqual(match?.remaining, "wait")
    }

    /// The homophone fold works inside normalization, not just at the tail.
    func testNormalizationFoldsCaseAndPunctuation() {
        // "ONIT!!! STOP???" must match the same as "onit stop".
        XCTAssertEqual(VoiceCommand.consume(from: "ONIT!!! STOP???")?.command, .stop)
    }
}
