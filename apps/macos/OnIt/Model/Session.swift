import Foundation
import Observation

/// One conversation, with its own stream and its own in-flight task.
///
/// Concurrency is the whole point of this type. A run in session A keeps
/// streaming while the user reads session B, because each `Session` owns its
/// own `Task` and nothing cancels it on selection change. The backend already
/// models this — `ApiSession.processing` is per session — so the client only
/// has to not get in the way.
@MainActor
@Observable
final class Session: Identifiable, Equatable {
    /// Identity equality. Two Session objects are the same session when they
    /// are the same object — the tests and the list diffing compare identity,
    /// not transcript contents.
    nonisolated static func == (lhs: Session, rhs: Session) -> Bool { lhs === rhs }

    let id: String
    let client: OnItClient
    /// Reply speech, nil when the server offers no TTS endpoint. Weak: the
    /// center is owned by AppModel and outlives any session.
    weak var tts: TtsCenter?

    var title: String
    var messages: [ChatMessage] = []
    var isProcessing = false
    var isLoaded = false
    var pendingApproval: ApprovalRequest?
    var loadError: String?
    /// When the server last touched this session, for the recency grouping.
    var lastActivity: Date?
    /// The list-row preview the server reported, used until this client has
    /// live messages of its own — a rehydrated session otherwise shows blank.
    var summaryPreview: String?

    private var streamTask: Task<Void, Never>?
    /// Internal, not private: the approval card reads the absolute deadline
    /// (it must recompute the count every second) and the tests drive the
    /// event path through it without a live server.
    var approvalDeadline: Date?

    init(id: String, title: String, client: OnItClient, lastActivity: Date? = nil, summaryPreview: String? = nil) {
        self.id = id
        self.title = title
        self.client = client
        self.lastActivity = lastActivity
        self.summaryPreview = summaryPreview
    }

    // MARK: - History

    /// Rehydrate from the server's JSONL transcript.
    ///
    /// The server returns only `{role, content}` pairs — the reasoning, tool
    /// steps and timings of past turns are not persisted in a form it replays.
    /// So a restored transcript is flatter than a live one, and that is
    /// honest: it shows what was said, not what the app happened to have in
    /// memory when it was said.
    func load() async {
        guard !isLoaded else { return }
        do {
            let history = try await client.history(session: id)
            messages = history.messages.map { entry in
                var message = ChatMessage(role: Self.role(from: entry.role), content: entry.content)
                // The server attaches descriptors for files the turn
                // produced; without them a restored transcript loses every
                // attachment and its Play button.
                if entry.role == "assistant", let files = entry.files, !files.isEmpty {
                    message.files = files
                }
                return message
            }
            isProcessing = history.processing
            isLoaded = true
        } catch {
            loadError = error.localizedDescription
            isLoaded = true
        }
    }

    /// The server's role string, as `onit/ui/api.py` writes it. System lines
    /// are their own thing — a notice, not an answer — and collapsing them
    /// into assistant bubbles would put words in the model's mouth.
    static func role(from serverRole: String) -> ChatMessage.Role {
        switch serverRole {
        case "user": return .user
        case "system": return .system
        default: return .assistant
        }
    }

    // MARK: - Sending

    func send(_ text: String, files: [String] = []) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty || !files.isEmpty, !isProcessing else { return }

        messages.append(ChatMessage(role: .user, content: trimmed))
        let answerIndex = messages.count
        messages.append(ChatMessage(role: .assistant, content: "", isStreaming: true))
        isProcessing = true
        pendingApproval = nil

        streamTask = Task { [weak self] in
            guard let self else { return }
            do {
                for try await event in client.chat(session: id, message: trimmed, files: files) {
                    apply(event, at: answerIndex)
                }
                finish(at: answerIndex)
            } catch {
                fail(error.localizedDescription, at: answerIndex)
            }
        }
    }

    func stop() {
        streamTask?.cancel()
        streamTask = nil
        Task { try? await client.stop(session: id) }
        isProcessing = false
        pendingApproval = nil
        if let last = messages.indices.last, messages[last].isStreaming {
            messages[last].isStreaming = false
            if messages[last].isEmpty { messages[last].content = "Stopped." }
        }
    }

    // MARK: - Approvals

    func answer(_ approval: ApprovalRequest, decision: String) {
        pendingApproval = nil
        approvalDeadline = nil
        Task { try? await client.approve(session: id, approvalID: approval.id, decision: decision) }
    }

    /// Seconds left on the current approval, or nil when none is pending.
    ///
    /// The card counts down and lets the server's deadline win. An approval
    /// that expires is a deny, and the UI must show it expiring rather than
    /// pretend the question is still open.
    func secondsRemaining(now: Date = Date()) -> Int? {
        ApprovalCard.remaining(until: approvalDeadline, now: now)
    }

    // MARK: - Event application

    /// Internal, not private: the tests drive this directly, which is the
    /// cheapest honest way to cover the event contract without a server.
    func apply(_ event: ChatEvent, at index: Int) {
        guard messages.indices.contains(index) else { return }
        switch event {
        case let .token(delta):
            messages[index].content += delta
            messages[index].status = nil

        case let .think(delta):
            messages[index].reasoning += delta

        case let .phaseEnd(content):
            if !content.isEmpty { messages[index].steps.append(content) }

        case let .status(text):
            messages[index].status = text
            if !text.isEmpty, messages[index].steps.last != text {
                messages[index].steps.append(text)
            }

        case .answerStart:
            messages[index].status = nil

        case let .approval(request):
            pendingApproval = request
            approvalDeadline = Date().addingTimeInterval(request.expiresIn)
            messages[index].steps.append("Waiting for approval: \(request.command)")

        case .approvalClosed:
            pendingApproval = nil
            approvalDeadline = nil

        case let .correction(correction):
            messages[index].correction = correction

        case let .done(payload):
            messages[index].content = payload.content
            messages[index].files = payload.files
            messages[index].elapsed = payload.elapsed
            messages[index].turn = payload.turn
            messages[index].ratingEnabled = payload.ratingEnabled
            messages[index].timing = payload.timing
            messages[index].isStreaming = false
            // The progress is scaffolding for the wait, not part of the
            // answer: the moment the answer lands it is erased, the same
            // way the browser folds the working away.
            messages[index].status = nil
            messages[index].steps = []
            isProcessing = false
            pendingApproval = nil
            approvalDeadline = nil
            if title.isEmpty || title == "New session" {
                title = String(messages[max(0, index - 1)].content.prefix(60))
            }
            // The answer is complete and on screen; now it may be heard.
            tts?.speak(payload.content, client: client, session: id, ttsId: payload.ttsId)

        case let .error(message):
            messages[index].error = message
            messages[index].isStreaming = false
            isProcessing = false

        case .unknown:
            break
        }
    }

    private func finish(at index: Int) {
        guard messages.indices.contains(index) else { return }
        messages[index].isStreaming = false
        messages[index].status = nil
        isProcessing = false
        pendingApproval = nil
    }

    private func fail(_ message: String, at index: Int) {
        guard messages.indices.contains(index) else { return }
        messages[index].error = message
        messages[index].isStreaming = false
        isProcessing = false
        pendingApproval = nil
    }

    // MARK: - Rating

    func rate(_ value: Int?) {
        guard let last = messages.indices.last(where: { messages[$0].role == .assistant }),
              let turn = messages[last].turn else { return }
        messages[last].rating = value
        Task { try? await client.rate(session: id, turn: turn, rating: value) }
    }
}
