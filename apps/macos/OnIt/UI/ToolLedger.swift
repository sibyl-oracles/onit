import SwiftUI

/// The folded list of what the agent did on its way to an answer.
///
/// Collapsed by default and one line tall, because the steps are evidence, not
/// the answer. Expanding shows them in order — the same order the server
/// emitted them, which is the only order that makes the run legible.
struct ToolLedger: View {
    let steps: [String]
    @State private var isExpanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Button {
                withAnimation(.onit) { isExpanded.toggle() }
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                        .font(.system(size: 9, weight: .semibold))
                    Text(summary)
                        .font(Typography.caption)
                }
                .foregroundStyle(Palette.tertiaryText)
            }
            .buttonStyle(.plain)

            if isExpanded {
                VStack(alignment: .leading, spacing: 3) {
                    ForEach(Array(steps.enumerated()), id: \.offset) { _, step in
                        HStack(alignment: .top, spacing: 6) {
                            Circle()
                                .fill(Palette.tertiaryText)
                                .frame(width: 3, height: 3)
                                .padding(.top, 6)
                            Text(step)
                                .font(Typography.mono)
                                .foregroundStyle(Palette.secondaryText)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }
                .padding(.leading, 13)
            }
        }
    }

    private var summary: String {
        steps.count == 1 ? "1 step" : "\(steps.count) steps"
    }
}
