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
    /// The status line the agent last reported, e.g. "Running shell".
    var status: String?
    /// Steps the agent took, in order, for the folded ledger.
    var steps: [String] = []

    var isEmpty: Bool { content.isEmpty && reasoning.isEmpty }
}
