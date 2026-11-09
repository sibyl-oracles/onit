import XCTest
@testable import OnIt

/// The session event reducer, driven directly — the cheapest honest way to
/// cover the event contract without a live server.
@MainActor
final class SessionTests: XCTestCase {
    private func makeSession() -> Session {
        // The base URL is never touched: these tests apply events, they do
        // not make requests.
        Session(
            id: "test-session",
            title: "New session",
            client: OnItClient(baseURL: URL(string: "http://127.0.0.1:9")!)
        )
    }

    /// The two-message shape `send()` builds, without the network call.
    private func openTurn(on session: Session) -> Int {
        session.messages.append(ChatMessage(role: .user, content: "list the files"))
        session.messages.append(ChatMessage(role: .assistant, content: "", isStreaming: true))
        session.isProcessing = true
        return session.messages.count - 1
    }

    // MARK: - Tokens and thinking

    func testTokenAppendsAndClearsStatus() {
        let session = makeSession()
        let index = openTurn(on: session)
        session.messages[index].status = "Running shell"

        session.apply(.token("Hel"), at: index)
        session.apply(.token("lo"), at: index)

        XCTAssertEqual(session.messages[index].content, "Hello")
        XCTAssertNil(session.messages[index].status)
        XCTAssertTrue(session.messages[index].isStreaming)
        XCTAssertTrue(session.isProcessing)
    }

    func testThinkGoesToReasoningNotContent() {
        let session = makeSession()
        let index = openTurn(on: session)

        session.apply(.think("pondering"), at: index)

        XCTAssertEqual(session.messages[index].reasoning, "pondering")
        XCTAssertEqual(session.messages[index].content, "")
    }

    func testAnswerStartClearsStatus() {
        let session = makeSession()
        let index = openTurn(on: session)
        session.messages[index].status = "Thinking"

        session.apply(.answerStart, at: index)

        XCTAssertNil(session.messages[index].status)
    }

    // MARK: - Status and phases

    func testStatusAppendsToStepsWithoutDuplicates() {
        let session = makeSession()
        let index = openTurn(on: session)

        session.apply(.status("Running shell"), at: index)
        session.apply(.status("Running shell"), at: index)
        session.apply(.status("Reading file"), at: index)

        XCTAssertEqual(session.messages[index].steps, ["Running shell", "Reading file"])
        XCTAssertEqual(session.messages[index].status, "Reading file")
    }

    func testPhaseEndAppendsNonEmptyContent() {
        let session = makeSession()
        let index = openTurn(on: session)

        session.apply(.phaseEnd("listed 3 files"), at: index)
        session.apply(.phaseEnd(""), at: index)

        XCTAssertEqual(session.messages[index].steps, ["listed 3 files"])
    }

    // MARK: - Approvals

    /// The auto-scroll trigger observes `activityFootprint`, not `content`.
    /// This is the regression it exists for: during a task the visible growth
    /// is reasoning, steps and status while `content` stays empty until the
    /// answer starts — a content-only trigger left the working phase
    /// unscrolled.
    func testActivityFootprintChangesWhileContentIsStillEmpty() {
        let session = makeSession()
        let index = openTurn(on: session)
        let idle = session.messages[index].activityFootprint

        session.apply(.think("weighing options"), at: index)
        let afterThink = session.messages[index].activityFootprint
        XCTAssertNotEqual(idle, afterThink)
        XCTAssertTrue(session.messages[index].content.isEmpty)

        session.apply(.status("Running shell"), at: index)
        XCTAssertNotEqual(afterThink, session.messages[index].activityFootprint)

        session.apply(.phaseEnd("listed 3 files"), at: index)
        XCTAssertNotEqual(afterThink, session.messages[index].activityFootprint)
    }

    /// Equal state must produce an equal footprint, or `onChange` fires on
    /// every render and each firing is a scroll layout pass.
    func testActivityFootprintIsStableWhenNothingGrows() {
        var message = ChatMessage(role: .assistant, content: "partial answer")
        message.reasoning = "thinking"
        message.steps = ["Running shell"]
        message.status = "Running shell"

        XCTAssertEqual(message.activityFootprint, message.activityFootprint)

        var same = message
        same.steps.append("Running shell")
        message.steps.append("Running shell")
        XCTAssertEqual(message.activityFootprint, same.activityFootprint)
    }

    func testApprovalSetsPendingAndDeadline() {
        let session = makeSession()
        let index = openTurn(on: session)
        let request = ApprovalRequest(
            id: "ap-1", command: "rm -rf build", reason: "clean build dir",
            subjects: ["build"], expiresIn: 30
        )

        session.apply(.approval(request), at: index)

        XCTAssertEqual(session.pendingApproval, request)
        XCTAssertEqual(session.messages[index].steps.last, "Waiting for approval: rm -rf build")
        // The deadline is expiresIn away from now, within a second of
        // scheduling slop; the countdown must move as time passes — the
        // frozen-count bug this guards against showed a static number.
        let deadline = try! XCTUnwrap(session.approvalDeadline)
        XCTAssertEqual(deadline.timeIntervalSinceNow, 30, accuracy: 1.0)
        let remaining = session.secondsRemaining()
        XCTAssertTrue((29...30).contains(remaining!), "got \(remaining!)")
        XCTAssertEqual(
            session.secondsRemaining(now: deadline),
            0,
            "at the deadline nothing is left"
        )
    }

    func testApprovalClosedClearsPending() {
        let session = makeSession()
        let index = openTurn(on: session)
        let request = ApprovalRequest(
            id: "ap-1", command: "ls", reason: "", subjects: [], expiresIn: 30
        )
        session.apply(.approval(request), at: index)

        session.apply(.approvalClosed("ap-1"), at: index)

        XCTAssertNil(session.pendingApproval)
        XCTAssertNil(session.approvalDeadline)
        XCTAssertNil(session.secondsRemaining())
    }

    func testSecondsRemainingFloorsAtZero() {
        let session = makeSession()
        session.approvalDeadline = Date().addingTimeInterval(-10)
        XCTAssertEqual(session.secondsRemaining(), 0)
    }

    // MARK: - Done

    func testDoneReplacesContentAndFinishesTurn() {
        let session = makeSession()
        let index = openTurn(on: session)
        session.apply(.token("partial"), at: index)

        let payload = DonePayload(
            content: "The full answer",
            files: [],
            elapsed: 1.5,
            tokensPerSecond: 42,
            turn: 3,
            ratingEnabled: true,
            timing: Timing(
                turns: 1, toolCalls: 2, modelSeconds: 1.1,
                promptTokensTotal: 100, completionTokens: 50, tokenStats: "150 tok"
            )
        )
        session.apply(.done(payload), at: index)

        XCTAssertEqual(session.messages[index].content, "The full answer")
        XCTAssertFalse(session.messages[index].isStreaming)
        XCTAssertFalse(session.isProcessing)
        XCTAssertNil(session.pendingApproval)
        XCTAssertEqual(session.messages[index].turn, 3)
        XCTAssertTrue(session.messages[index].ratingEnabled)
    }

    func testDoneTitlesTheSessionFromTheUserMessage() {
        let session = makeSession()
        let index = openTurn(on: session)
        let payload = DonePayload(
            content: "done", files: [], elapsed: 0, tokensPerSecond: 0,
            turn: 1, ratingEnabled: false,
            timing: Timing(turns: 0, toolCalls: 0, modelSeconds: 0,
                           promptTokensTotal: 0, completionTokens: 0, tokenStats: "")
        )
        session.apply(.done(payload), at: index)
        XCTAssertEqual(session.title, "list the files")
    }

    // MARK: - Errors and unknowns

    func testErrorMarksTheTurn() {
        let session = makeSession()
        let index = openTurn(on: session)

        session.apply(.error("boom"), at: index)

        XCTAssertEqual(session.messages[index].error, "boom")
        XCTAssertFalse(session.messages[index].isStreaming)
        XCTAssertFalse(session.isProcessing)
    }

    func testUnknownEventChangesNothing() {
        let session = makeSession()
        let index = openTurn(on: session)
        let before = session.messages[index]

        session.apply(.unknown(name: "new_thing", data: "{}"), at: index)

        XCTAssertEqual(session.messages[index], before)
        XCTAssertTrue(session.isProcessing)
    }

    func testEventForMissingMessageIsIgnored() {
        let session = makeSession()
        // No messages at all: the reducer must not crash on a stale index.
        session.apply(.token("orphan"), at: 5)
        XCTAssertTrue(session.messages.isEmpty)
    }

    // MARK: - History roles

    func testRoleMapping() {
        XCTAssertEqual(Session.role(from: "user"), .user)
        XCTAssertEqual(Session.role(from: "assistant"), .assistant)
        XCTAssertEqual(Session.role(from: "system"), .system)
        XCTAssertEqual(Session.role(from: "tool"), .assistant, "unknown roles stay out of the user lane")
    }
}
