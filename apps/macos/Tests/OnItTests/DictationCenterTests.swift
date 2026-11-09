import XCTest
@testable import OnIt

@MainActor
final class DictationCenterTests: XCTestCase {
    /// The accumulator: a finalized utterance becomes the base, and the
    /// in-flight hypothesis rides alongside it. One breath between sentences
    /// finalizes the first — the join must still read as one dictation.
    func testFinalAndPartialJoin() {
        let center = DictationCenter()
        center.commitFinal("what should")
        center.updatePartial("what should onit do")
        XCTAssertEqual(center.currentText, "what should what should onit do")
        center.commitFinal("what should onit do")
        XCTAssertEqual(center.currentText, "what should onit do")
    }

    /// A final replaces the in-flight hypothesis it belongs to.
    func testFinalReplacesPartial() {
        let center = DictationCenter()
        center.updatePartial("hello wor")
        center.commitFinal("hello world")
        XCTAssertEqual(center.currentText, "hello world")
    }

    /// The task must not wipe the task — this is the reported overwrite bug.
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

    /// A final that shares only a prefix with the transcript is a new
    /// utterance, not a repeat — it appends.
    func testPrefixSharedFinalAppends() {
        let center = DictationCenter()
        center.commitFinal("run the tests")
        center.commitFinal("run the linter too")
        XCTAssertEqual(center.currentText, "run the tests run the linter too")
    }

    func testResetClearsAccumulator() {
        let center = DictationCenter()
        center.commitFinal("first pass")
        center.updatePartial("second")
        center.reset()
        XCTAssertEqual(center.currentText, "")
    }

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
        XCTAssertNil(snap.pendingCommand)
        XCTAssertEqual(snap.currentText, "")

        center.commitFinal("note to self")
        snap = center.snapshot
        XCTAssertEqual(snap.currentText, "note to self")
    }

    // MARK: - The reported overwrite: a command firing must not resurrect
    // the words it consumed, and the text it leaves must be the raw task.

    /// The user's exact flow: dictate the task, pause (it finalizes), say
    /// "run". The bare command fires on the finalized utterance — and the
    /// words it consumed are stripped from the raw text, so what is left to
    /// send is the task exactly as dictated.
    func testBareRunAfterPauseFiresAndStripsRaw() {
        let center = DictationCenter()
        center.commitFinal("Summarize chapter 3.")
        XCTAssertNil(center.snapshot.pendingCommand)
        center.commitFinal("run.")
        XCTAssertEqual(center.snapshot.pendingCommand?.command, .run)
        XCTAssertEqual(center.takeCommand(), .run)
        XCTAssertEqual(center.currentText, "Summarize chapter 3.")
    }

    /// "go" works the same way — the user's second requested trigger.
    func testBareGoAfterPauseFires() {
        let center = DictationCenter()
        center.commitFinal("summarize chapter 3")
        center.commitFinal("go")
        XCTAssertEqual(center.snapshot.pendingCommand?.command, .run)
    }

    /// "execute" as its own utterance after the pause also fires.
    func testBareExecuteAfterPauseFires() {
        let center = DictationCenter()
        center.commitFinal("summarize chapter 3")
        center.commitFinal("execute")
        XCTAssertEqual(center.snapshot.pendingCommand?.command, .run)
    }

    /// The anchored form fires on a partial — a barge-in cannot wait for a
    /// pause — and the strip keeps the raw task text. After a final, the
    /// in-flight hypothesis holds only the new utterance's words.
    func testAnchoredRunFiresOnPartialAndStripsRaw() {
        let center = DictationCenter()
        center.commitFinal("Summarize chapter 3.")
        center.updatePartial("Onit, run.")
        XCTAssertEqual(center.currentText, "Summarize chapter 3. Onit, run.")
        XCTAssertEqual(center.snapshot.pendingCommand?.command, .run)
        XCTAssertEqual(center.takeCommand(), .run)
        XCTAssertEqual(center.currentText, "Summarize chapter 3.")
    }

    /// A spoken stop strips its words but keeps the task text, so the next
    /// "onit, run" still sends the whole task.
    func testStopStripsButKeepsTaskText() {
        let center = DictationCenter()
        center.commitFinal("Summarize chapter 3.")
        center.commitFinal("Onit, stop.")
        XCTAssertEqual(center.snapshot.pendingCommand?.command, .stop)
        XCTAssertEqual(center.takeCommand(), .stop)
        XCTAssertEqual(center.currentText, "Summarize chapter 3.")
    }

    /// A bare stop with nothing behind it still fires — after a send the
    /// transcript is empty, and barge-in is exactly what is wanted.
    func testBareStopWithEmptyTranscriptFires() {
        let center = DictationCenter()
        center.commitFinal("stop")
        XCTAssertEqual(center.snapshot.pendingCommand?.command, .stop)
    }

    /// A bare command never fires mid-stream: on a partial there is no
    /// utterance boundary, so "run" at the tail of a flowing sentence is
    /// task text until the user pauses.
    func testBareCommandDoesNotFireOnPartial() {
        let center = DictationCenter()
        center.commitFinal("summarize chapter 3")
        center.updatePartial("summarize chapter 3 run")
        XCTAssertNil(center.snapshot.pendingCommand)
    }

    /// A sentence that merely ends with a command word is one utterance,
    /// not an utterance plus a command — the pause is the separator.
    func testSentenceEndingInCommandWordIsTaskText() {
        let center = DictationCenter()
        center.commitFinal("yesterday I went for a run")
        XCTAssertNil(center.snapshot.pendingCommand)
        center.commitFinal("tell the build to stop")
        XCTAssertNil(center.snapshot.pendingCommand)
    }

    // MARK: - The pause watchdog: on-device recognition often emits no
    // mid-stream finals, so the bare form must unlock on a silence lull.

    /// The reported flow, in the shape the on-device path actually delivers
    /// it: the task as a final, "run" as a *partial* that never finalizes,
    /// then the pause. The watchdog commits the stalled hypothesis and the
    /// bare command fires — no recognizer final required.
    func testWatchdogFiresBareRunFromStalledPartial() {
        let center = DictationCenter()
        center.commitFinal("Summarize chapter 3.")
        center.updatePartial("run")
        XCTAssertNil(center.snapshot.pendingCommand) // no boundary yet
        center.commitStalledPartial() // the 1s lull
        XCTAssertEqual(center.snapshot.pendingCommand?.command, .run)
        XCTAssertEqual(center.takeCommand(), .run)
        XCTAssertEqual(center.currentText, "Summarize chapter 3.")
    }

    /// "go" through the same lull path.
    func testWatchdogFiresBareGoFromStalledPartial() {
        let center = DictationCenter()
        center.commitFinal("summarize chapter 3")
        center.updatePartial("go")
        center.commitStalledPartial()
        XCTAssertEqual(center.snapshot.pendingCommand?.command, .run)
    }

    /// The watchdog must not invent words: it cannot commit while the
    /// hypothesis is empty, and it cannot fire a second time on the same
    /// silence (it consumes its own commit).
    func testWatchdogNoOpWithoutPartial() {
        let center = DictationCenter()
        center.commitFinal("summarize chapter 3")
        center.commitStalledPartial()
        XCTAssertEqual(center.snapshot.currentText, "summarize chapter 3")
        center.commitStalledPartial()
        XCTAssertEqual(center.snapshot.currentText, "summarize chapter 3")
        XCTAssertNil(center.snapshot.pendingCommand)
    }

    /// A lull that is not a command just lands the words — a person thinking
    /// mid-task ("…chapter 3" … silence … keeps talking) must not lose text
    /// or fire anything.
    func testWatchdogCommitsPlainTaskTextOnLull() {
        let center = DictationCenter()
        center.updatePartial("summarize")
        center.commitStalledPartial()
        XCTAssertEqual(center.snapshot.currentText, "summarize")
        XCTAssertNil(center.snapshot.pendingCommand)
        // Continuing after the lull appends, exactly as after a real final.
        center.commitFinal("chapter 3")
        XCTAssertEqual(center.snapshot.currentText, "summarize chapter 3")
    }

    /// A real recognizer final arriving after the watchdog already committed
    /// the same words must not duplicate them: the final is the cumulative
    /// shape of what we committed, so it replaces.
    func testRealFinalAfterWatchdogCommitDoesNotDuplicate() {
        let center = DictationCenter()
        center.updatePartial("summarize chapter 3")
        center.commitStalledPartial()
        XCTAssertEqual(center.snapshot.currentText, "summarize chapter 3")
        center.commitFinal("Summarize chapter 3.")
        XCTAssertEqual(center.snapshot.currentText, "Summarize chapter 3.")
    }

    /// An anchored command fires mid-stream, before any lull — the watchdog
    /// must not delay or interfere with the barge-in path.
    func testAnchoredCommandStillFiresBeforeWatchdog() {
        let center = DictationCenter()
        center.commitFinal("Summarize chapter 3.")
        center.updatePartial("onit run")
        XCTAssertEqual(center.snapshot.pendingCommand?.command, .run)
        // The watchdog pending underneath must not double-fire anything.
        center.commitStalledPartial()
        XCTAssertEqual(center.takeCommand(), .run)
    }

    /// A new task that begins with a command word must not fire: the bare
    /// form needs a task behind it, and "run the linter too" is a task.
    func testBareCommandNeedsWholeUtterance() {
        let center = DictationCenter()
        center.commitFinal("summarize chapter 3")
        center.commitFinal("run the linter too")
        XCTAssertNil(center.snapshot.pendingCommand)
    }

    /// The server path repeats everything so far; the bare form must read
    /// the command from the delta past the previous transcript.
    func testBareCommandOnCumulativeFinal() {
        let center = DictationCenter()
        center.commitFinal("summarize chapter 3")
        center.commitFinal("summarize chapter 3 go")
        XCTAssertEqual(center.snapshot.pendingCommand?.command, .run)
        XCTAssertEqual(center.takeCommand(), .run)
        XCTAssertEqual(center.currentText, "summarize chapter 3")
    }

    /// Consuming is once-only: the second take returns nil.
    func testTakeCommandIsOnceOnly() {
        let center = DictationCenter()
        center.commitFinal("summarize chapter 3")
        center.commitFinal("run")
        XCTAssertEqual(center.takeCommand(), .run)
        XCTAssertNil(center.takeCommand())
        XCTAssertNil(center.snapshot.pendingCommand)
    }

    /// A send clears the pending command with everything else.
    func testResetClearsPendingCommand() {
        let center = DictationCenter()
        center.commitFinal("summarize chapter 3")
        center.commitFinal("run")
        XCTAssertNotNil(center.snapshot.pendingCommand)
        center.reset()
        XCTAssertNil(center.snapshot.pendingCommand)
        XCTAssertEqual(center.currentText, "")
    }
}

@MainActor
final class VoiceCommandTests: XCTestCase {
    // MARK: - "onit, run" / "onit, execute" / "onit, go"

    /// The plain execute command, with and without punctuation, in both
    /// spellings of the anchor.
    func testOnitRun() {
        XCTAssertEqual(VoiceCommand.consume(from: "onit run")?.command, .run)
        XCTAssertEqual(VoiceCommand.consume(from: "Onit, run.")?.command, .run)
        XCTAssertEqual(VoiceCommand.consume(from: "on it run")?.command, .run)
        XCTAssertEqual(VoiceCommand.consume(from: "onit execute")?.command, .run)
        XCTAssertEqual(VoiceCommand.consume(from: "onit go")?.command, .run)
    }

    func testOnitRunIt() {
        XCTAssertEqual(VoiceCommand.consume(from: "onit run it")?.command, .run)
        XCTAssertEqual(VoiceCommand.consume(from: "on it, run it")?.command, .run)
    }

    func testOnitRunLeavesTaskText() {
        let match = VoiceCommand.consume(from: "summarize this chapter, onit run")
        XCTAssertEqual(match?.command, .run)
        XCTAssertEqual(match?.remaining, "summarize this chapter")
        XCTAssertEqual(match?.consumed, 2)
    }

    func testOnitRunWithFillers() {
        XCTAssertEqual(VoiceCommand.consume(from: "onit, um, execute")?.command, .run)
        XCTAssertEqual(VoiceCommand.consume(from: "hey onit please run")?.command, .run)
    }

    // MARK: - "onit, stop"

    func testOnitStop() {
        XCTAssertEqual(VoiceCommand.consume(from: "onit stop")?.command, .stop)
        XCTAssertEqual(VoiceCommand.consume(from: "on it stop")?.command, .stop)
        XCTAssertEqual(VoiceCommand.consume(from: "Onit, stop.")?.command, .stop)
        XCTAssertEqual(VoiceCommand.consume(from: "onit stop it")?.command, .stop)
    }

    /// Bare "execute", "run it", "stop" are ordinary words that appear in
    /// half of all dictated tasks, and treating them as commands consumed
    /// the task — they only fire as their own utterance after a pause,
    /// which the `utterance` parameter unlocks.
    func testBareCommandsNeedAnUtteranceBoundary() {
        XCTAssertNil(VoiceCommand.consume(from: "run it"))
        XCTAssertNil(VoiceCommand.consume(from: "execute"))
        XCTAssertNil(VoiceCommand.consume(from: "stop it"))
        XCTAssertNil(VoiceCommand.consume(from: "stop"))
        XCTAssertNil(VoiceCommand.consume(from: "run it", utterance: "run it and more"))
        XCTAssertNil(VoiceCommand.consume(from: "tell the build to stop", utterance: "tell the build to stop"))
    }

    /// The pause-delimited form: the utterance after the pause is exactly
    /// the command phrase.
    func testBareCommandsFireOnUtterance() {
        XCTAssertEqual(VoiceCommand.consume(from: "summarize chapter 3 run", utterance: "run")?.command, .run)
        XCTAssertEqual(VoiceCommand.consume(from: "summarize chapter 3 go", utterance: "go")?.command, .run)
        XCTAssertEqual(VoiceCommand.consume(from: "summarize chapter 3 execute", utterance: "execute")?.command, .run)
        XCTAssertEqual(VoiceCommand.consume(from: "summarize chapter 3 run it", utterance: "run it")?.command, .run)
        XCTAssertEqual(VoiceCommand.consume(from: "summarize chapter 3 stop", utterance: "stop")?.command, .stop)
        XCTAssertEqual(VoiceCommand.consume(from: "summarize chapter 3 stop it", utterance: "stop it")?.command, .stop)
    }

    /// A bare command with nothing behind it fires with empty remaining —
    /// barge-in after a send, where the transcript is already clear.
    func testBareStopWithEmptyStream() {
        let match = VoiceCommand.consume(from: "stop", utterance: "stop")
        XCTAssertEqual(match?.command, .stop)
        XCTAssertEqual(match?.remaining, "")
        XCTAssertEqual(match?.consumed, 1)
    }

    /// A bare command whose words are not at the stream tail is stale — a
    /// newer partial is in flight, and it vetoes the fire.
    func testBareCommandVetoedByNewerPartial() {
        XCTAssertNil(VoiceCommand.consume(from: "summarize chapter 3 run the", utterance: "run"))
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

    /// Extra words after the anchor make it a task, not a command.
    func testExtraWordsAfterAnchorDoNotFire() {
        XCTAssertNil(VoiceCommand.consume(from: "onit run the tests"))
        XCTAssertNil(VoiceCommand.consume(from: "onit execute this script"))
        XCTAssertNil(VoiceCommand.consume(from: "onit stop the server"))
        XCTAssertNil(VoiceCommand.consume(from: "onit run it and then stop"))
    }

    // MARK: - Raw-text strip

    /// The strip removes exactly the command's raw tokens and keeps the
    /// task's casing and punctuation.
    func testStrippingLastWords() {
        XCTAssertEqual(
            VoiceCommand.strippingLastWords(2, from: "Summarize chapter 3. Onit, run."),
            "Summarize chapter 3."
        )
        XCTAssertEqual(
            VoiceCommand.strippingLastWords(1, from: "Summarize chapter 3. run."),
            "Summarize chapter 3."
        )
        // Filler tokens count as zero words and are cut with the command.
        XCTAssertEqual(
            VoiceCommand.strippingLastWords(2, from: "task onit, um, run"),
            "task"
        )
        // A stop phrase strips the same way — "on it" is two words.
        XCTAssertEqual(
            VoiceCommand.strippingLastWords(3, from: "wait, on it stop"),
            "wait"
        )
    }

    /// The strip refuses to guess when the raw tokens do not align — the
    /// caller keeps the text as-is rather than corrupt it.
    func testStrippingLastWordsRefusesMisalignment() {
        XCTAssertNil(VoiceCommand.strippingLastWords(0, from: "task"))
        XCTAssertNil(VoiceCommand.strippingLastWords(5, from: "two words"))
        // "on,it" is one raw token with two normalized words: cutting one
        // word out of it would corrupt the token, so the strip bails.
        XCTAssertNil(VoiceCommand.strippingLastWords(1, from: "task on,it"))
    }
}
