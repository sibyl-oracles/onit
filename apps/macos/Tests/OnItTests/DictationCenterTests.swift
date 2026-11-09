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

    /// On-device recognition finalizes per utterance: each final holds only
    /// the words since the last pause. A command spoken after a dictated
    /// task must not wipe the task — this is the reported overwrite bug.
    func testFinalAfterPauseAppends() {
        let center = DictationCenter()
        center.commitFinal("summarize chapter 3")
        center.commitFinal("execute")
        XCTAssertEqual(center.currentText, "summarize chapter 3 execute")
    }

    /// The server path repeats everything recognized so far; that shape
    /// must replace, not append on top of itself.
    func testCumulativeFinalReplaces() {
        let center = DictationCenter()
        center.commitFinal("summarize chapter 3")
        center.commitFinal("summarize chapter 3 execute")
        XCTAssertEqual(center.currentText, "summarize chapter 3 execute")
    }

    /// The cumulative test is case- and punctuation-insensitive: the
    /// recognizer's casing and punctuation are not stable across utterances.
    func testCumulativeFinalIgnoresCaseAndPunctuation() {
        let center = DictationCenter()
        center.commitFinal("Summarize chapter 3.")
        center.commitFinal("Summarize chapter 3. Execute!")
        XCTAssertEqual(center.currentText, "Summarize chapter 3. Execute!")
    }

    /// A final that only shares a prefix with the transcript is a new
    /// utterance, not a longer version of it — it must append.
    func testPrefixSharedFinalAppends() {
        let center = DictationCenter()
        center.commitFinal("run the tests")
        center.commitFinal("run the linter too")
        XCTAssertEqual(center.currentText, "run the tests run the linter too")
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

    /// Reset also clears the in-flight hypothesis, so a partial that arrives
    /// after the flush cannot resurrect already-sent words into the draft.
    func testResetDropsPartial() {
        let center = DictationCenter()
        center.updatePartial("summarize chapter 3 run it")
        center.reset()
        center.updatePartial("summarize chapter 3 run it")
        XCTAssertEqual(center.currentText, "summarize chapter 3 run it")
        center.commitFinal("summarize chapter 3 run it")
        center.reset()
        XCTAssertEqual(center.currentText, "")
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
    // MARK: - "onit, run" / "onit, execute"

    /// The plain execute command, with and without punctuation, in both
    /// spellings of the anchor.
    func testOnitRun() {
        XCTAssertEqual(VoiceCommand.consume(from: "onit run")?.command, .run)
        XCTAssertEqual(VoiceCommand.consume(from: "Onit, run.")?.command, .run)
        XCTAssertEqual(VoiceCommand.consume(from: "on it run")?.command, .run)
        XCTAssertEqual(VoiceCommand.consume(from: "onit execute")?.command, .run)
    }

    /// "onit, run it" is an accepted variant.
    func testOnitRunIt() {
        XCTAssertEqual(VoiceCommand.consume(from: "onit run it")?.command, .run)
        XCTAssertEqual(VoiceCommand.consume(from: "on it, run it")?.command, .run)
    }

    /// The command peels off the tail, leaving the task text behind.
    func testOnitRunLeavesTaskText() {
        let match = VoiceCommand.consume(from: "summarize this chapter, onit run")
        XCTAssertEqual(match?.command, .run)
        XCTAssertEqual(match?.remaining, "summarize this chapter")
    }

    /// Filler words must not block the match.
    func testOnitRunWithFillers() {
        XCTAssertEqual(VoiceCommand.consume(from: "onit, um, execute")?.command, .run)
        XCTAssertEqual(VoiceCommand.consume(from: "hey onit please run")?.command, .run)
    }

    // MARK: - "onit, stop" barge-in

    /// Both spellings of the anchor: the recognizer is inconsistent about
    /// whether "onit" comes back as one word or the English phrase.
    func testOnitStop() {
        XCTAssertEqual(VoiceCommand.consume(from: "onit stop")?.command, .stop)
        XCTAssertEqual(VoiceCommand.consume(from: "on it stop")?.command, .stop)
        XCTAssertEqual(VoiceCommand.consume(from: "Onit, stop.")?.command, .stop)
        XCTAssertEqual(VoiceCommand.consume(from: "onit stop it")?.command, .stop)
    }

    /// Bare "execute", "run it", and "stop it" are deliberately not
    /// commands: without the anchor they are ordinary words that end half
    /// of all dictated tasks, and treating them as commands consumed the
    /// task instead of executing it.
    func testBareCommandsAreNotCommands() {
        XCTAssertNil(VoiceCommand.consume(from: "run it"))
        XCTAssertNil(VoiceCommand.consume(from: "execute"))
        XCTAssertNil(VoiceCommand.consume(from: "stop it"))
        XCTAssertNil(VoiceCommand.consume(from: "stop"))
    }

    /// A task that merely mentions the command words must not fire — the
    /// command phrase must be the entire tail after the anchor.
    func testExtraWordsAfterAnchorDoNotFire() {
        XCTAssertNil(VoiceCommand.consume(from: "onit run the tests"))
        XCTAssertNil(VoiceCommand.consume(from: "onit execute this script"))
        XCTAssertNil(VoiceCommand.consume(from: "onit stop the server"))
        XCTAssertNil(VoiceCommand.consume(from: "onit run it and then stop"))
    }

    /// A task addressed to onit mid-sentence passes through untouched.
    func testMidSentenceMentionDoesNotFire() {
        XCTAssertNil(VoiceCommand.consume(from: "how do I run it in production"))
        XCTAssertNil(VoiceCommand.consume(from: "tell onit to stop the build"))
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

    /// The last anchor wins: a task that quotes an earlier command still
    /// ends with the words the user just said.
    func testLastAnchorWins() {
        let match = VoiceCommand.consume(from: "onit run, no wait, onit stop")
        XCTAssertEqual(match?.command, .stop)
        XCTAssertEqual(match?.remaining, "onit run no wait")
    }
}
