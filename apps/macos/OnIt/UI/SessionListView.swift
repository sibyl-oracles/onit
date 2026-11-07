import SwiftUI

/// The session list, grouped by recency.
///
/// Grouping is by when a person last used a session, not by when it was
/// created: "Yesterday" is a memory, "created 14:02" is a record.
struct SessionListView: View {
    @Environment(AppModel.self) private var model
    @State private var renaming: Session?
    @State private var renameText = ""

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider().overlay(Palette.hairline)

            if model.sessions.isEmpty {
                VStack(spacing: 8) {
                    Text("No sessions yet")
                        .font(Typography.body)
                        .foregroundStyle(Palette.secondaryText)
                    Text("Start one with ⌘N")
                        .font(Typography.caption)
                        .foregroundStyle(Palette.tertiaryText)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                List(selection: selection) {
                    ForEach(model.grouped(model.sessions)) { group in
                        Section {
                            ForEach(group.sessions) { session in
                                SessionRow(session: session)
                                    .tag(session.id)
                                    .contextMenu {
                                        Button("Rename…") {
                                            renameText = session.title
                                            renaming = session
                                        }
                                        Divider()
                                        Button("Delete", role: .destructive) {
                                            Task { await model.delete(session) }
                                        }
                                    }
                            }
                        } header: {
                            Text(group.id)
                                .font(Typography.caption)
                                .foregroundStyle(Palette.tertiaryText)
                        }
                    }
                }
                .listStyle(.sidebar)
                .scrollContentBackground(.hidden)
            }
        }
        .background(Palette.surface)
        .alert("Rename Session", isPresented: Binding(
            get: { renaming != nil },
            set: { if !$0 { renaming = nil } }
        )) {
            TextField("Name", text: $renameText)
            Button("Cancel", role: .cancel) { renaming = nil }
            Button("Rename") {
                if let session = renaming {
                    Task { await model.rename(session, to: renameText) }
                }
                renaming = nil
            }
        }
    }

    private var selection: Binding<Session.ID?> {
        Binding(
            get: { model.selectedSessionID },
            set: { newValue in
                model.selectedSessionID = newValue
                if let session = model.selectedSession {
                    Task { await session.load() }
                }
            }
        )
    }

    private var header: some View {
        HStack {
            Text(model.selectedConnection?.name ?? "OnIt")
                .font(Typography.title)
                .lineLimit(1)
            Spacer()
            Button {
                Task { await model.newSession() }
            } label: {
                Image(systemName: "square.and.pencil")
                    .font(.system(size: 13))
            }
            .buttonStyle(.plain)
            .help("New session (⌘N)")
            .accessibilityLabel("New session")
        }
        .padding(.horizontal, Metrics.gutter)
        .padding(.vertical, 12)
    }
}

private struct SessionRow: View {
    let session: Session

    var body: some View {
        HStack(spacing: 8) {
            if session.isProcessing {
                ProgressView()
                    .controlSize(.mini)
                    .frame(width: 12)
            } else if session.pendingApproval != nil {
                Image(systemName: "exclamationmark.circle.fill")
                    .font(.system(size: 11))
                    .foregroundStyle(Color.orange)
                    .frame(width: 12)
            }

            VStack(alignment: .leading, spacing: 1) {
                Text(session.title)
                    .font(Typography.body)
                    .lineLimit(1)
                if let last = session.messages.last, last.role == .assistant, !last.content.isEmpty {
                    Text(last.content)
                        .font(Typography.caption)
                        .foregroundStyle(Palette.tertiaryText)
                        .lineLimit(1)
                } else if let preview = session.summaryPreview, !preview.isEmpty {
                    // A rehydrated session has no live transcript yet; the
                    // server's preview is what the row showed before.
                    Text(preview)
                        .font(Typography.caption)
                        .foregroundStyle(Palette.tertiaryText)
                        .lineLimit(1)
                }
            }
        }
        .padding(.vertical, 2)
    }
}
