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
