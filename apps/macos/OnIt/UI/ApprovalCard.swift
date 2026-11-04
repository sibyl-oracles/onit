import SwiftUI

/// A gated command, inline in the transcript.
///
/// Not a sheet. A sheet covers the transcript, and the transcript is exactly
/// the context the decision depends on — the user needs to see what the agent
/// was doing when it decided to run this. The card sits next to the turn it
/// gates instead.
///
/// Three properties, all enforced by the backend and preserved here:
///
/// - **Session-scoped.** The answer goes back on this session's id, never a
///   global one.
/// - **Fail-closed.** Deny is the default button and Escape is its shortcut.
///   A timeout, a closed window or no answer at all is a deny server-side; the
///   card counts down and disappears rather than pretending the question is
///   still open.
/// - **Expiring.** `expiresIn` is the server's own budget.
struct ApprovalCard: View {
    let approval: ApprovalRequest
    let secondsRemaining: Int?
    let onDecision: (String) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 6) {
                Image(systemName: "exclamationmark.shield")
                    .font(.system(size: 12))
                    .foregroundStyle(Color.orange)
                Text("OnIt wants to run a command")
                    .font(Typography.bodyEmphasis)
                Spacer()
                if let secondsRemaining {
                    Text("\(secondsRemaining)s")
                        .font(Typography.caption.monospacedDigit())
                        .foregroundStyle(secondsRemaining <= 10 ? Palette.danger : Palette.tertiaryText)
                }
            }

            Text(approval.command)
                .font(Typography.mono)
                .textSelection(.enabled)
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Palette.surface, in: RoundedRectangle(cornerRadius: Metrics.controlRadius))

            if !approval.reason.isEmpty {
                Text(approval.reason)
                    .font(Typography.caption)
                    .foregroundStyle(Palette.secondaryText)
                    .fixedSize(horizontal: false, vertical: true)
            }

            if !approval.subjects.isEmpty {
                Text(approval.subjects.joined(separator: ", "))
                    .font(Typography.caption)
                    .foregroundStyle(Palette.tertiaryText)
                    .lineLimit(2)
            }

            HStack(spacing: 8) {
                Spacer()
                Button("Deny") { onDecision("deny") }
                    .keyboardShortcut(.cancelAction)
                Button("Always Allow") { onDecision("session") }
                Button("Allow Once") { onDecision("once") }
                    .keyboardShortcut(.defaultAction)
            }
            .controlSize(.small)
        }
        .padding(12)
        .background(Color.orange.opacity(0.07), in: RoundedRectangle(cornerRadius: Metrics.cardRadius))
        .overlay {
            RoundedRectangle(cornerRadius: Metrics.cardRadius)
                .stroke(Color.orange.opacity(0.35), lineWidth: 1)
        }
    }
}
