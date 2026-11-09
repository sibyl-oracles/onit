import Foundation

/// One turn's worth of conversation, as the transcript renders it.
///
/// `reasoning` is kept apart from `content` on purpose. The backend streams
/// them on two channels so a client can fold the thinking away when the answer
/// lands; merging them into one string here would make that impossible.
struct ChatMessage: Identifiable, Sendable, Equatable {
    enum Role: Sendable { case user, assistant, system }

    let id = UUID()
    let role: Role
    var content: String
    var reasoning: String = ""
    var files: [FileInfo] = []
    var isStreaming: Bool = false
    var error: String?
    var correction: Correction?
    var elapsed: Double?
    var turn: Int?
    var ratingEnabled: Bool = false
    var rating: Int?
    /// Where the run's time and tokens went, as the server reported them on
    /// `done`. Nil for restored messages: history does not persist it, and a
    /// blank row would claim a measurement that was never taken.
    var timing: Timing?
    /// The status line the agent last reported, e.g. "Running shell".
    var status: String?
    /// Steps the agent took, in order, for the folded ledger.
    var steps: [String] = []

    /// The growth of one streaming message, as plain numbers. Equatable so
    /// SwiftUI's `onChange` can compare it without copying the text itself.
    struct ActivityFootprint: Equatable {
        var contentBytes: Int
        var reasoningBytes: Int
        var steps: Int
        var status: String?
    }

    var isEmpty: Bool { content.isEmpty && reasoning.isEmpty }

    /// What can grow while this message is being written, as O(1) counts
    /// rather than a copy of the text. The transcript's auto-scroll triggers
    /// on this and it is observed for every token, so building it must not
    /// allocate. It deliberately covers more than `content`: during a task
    /// the visible growth is the reasoning, the step ledger and the status
    /// line, while `content` stays empty until the answer starts — a
    /// `content`-only trigger would leave the whole working phase unscrolled.
    var activityFootprint: ActivityFootprint {
        ActivityFootprint(
            contentBytes: content.utf8.count,
            reasoningBytes: reasoning.utf8.count,
            steps: steps.count,
            status: status
        )
    }
}
