import SwiftUI

/// The three-level shell: connection rail → session list → transcript.
///
/// The rail is a real column rather than a menu because a second connection is
/// the moment the app stops being "the local harness's window" and becomes a
/// client. Retrofitting that later would mean touching the session list, the
/// settings and the sidecar lifecycle at once.
struct RootView: View {
    @Environment(AppModel.self) private var model
    @State private var columnVisibility: NavigationSplitViewVisibility = .all

    var body: some View {
        NavigationSplitView(columnVisibility: $columnVisibility) {
            ConnectionRail()
                .navigationSplitViewColumnWidth(
                    min: Metrics.railWidth, ideal: Metrics.railWidth, max: Metrics.railWidth
                )
        } content: {
            SessionListView()
                .navigationSplitViewColumnWidth(
                    min: 200, ideal: Metrics.sidebarWidth, max: 320
                )
        } detail: {
            DetailPane()
        }
        .navigationSplitViewStyle(.balanced)
        .alert(
            "Something went wrong",
            isPresented: Binding(
                get: { model.lastError != nil },
                set: { if !$0 { model.lastError = nil } }
            )
        ) {
            Button("OK", role: .cancel) { model.lastError = nil }
        } message: {
            Text(model.lastError ?? "")
        }
    }
}

private struct DetailPane: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        if let session = model.selectedSession {
            ChatView(session: session)
        } else if case let .failed(reason) = model.sidecar.state {
            SidecarFailureView(reason: reason)
        } else if !model.sidecar.state.isReady {
            StartingView()
        } else {
            EmptyDetailView()
        }
    }
}

private struct StartingView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        VStack(spacing: 14) {
            ProgressView()
                .controlSize(.small)
            Text("Starting the harness…")
                .font(Typography.body)
                .foregroundStyle(Palette.secondaryText)
            if !model.sidecar.logTail.isEmpty {
                ScrollView {
                    Text(model.sidecar.logTail.suffix(8).joined(separator: "\n"))
                        .font(Typography.mono)
                        .foregroundStyle(Palette.tertiaryText)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxWidth: 520, maxHeight: 120)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Palette.surface)
    }
}

private struct SidecarFailureView: View {
    @Environment(AppModel.self) private var model
    let reason: String

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "exclamationmark.triangle")
                .font(.system(size: 28))
                .foregroundStyle(Palette.danger)
            Text("The harness did not start")
                .font(Typography.title)
            Text(reason)
                .font(Typography.body)
                .foregroundStyle(Palette.secondaryText)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 420)

            if !model.sidecar.logTail.isEmpty {
                ScrollView {
                    Text(model.sidecar.logTail.joined(separator: "\n"))
                        .font(Typography.mono)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxWidth: 560, maxHeight: 200)
                .padding(10)
                .background(Palette.raised, in: RoundedRectangle(cornerRadius: Metrics.controlRadius))
            }

            Button("Try Again") {
                Task { await model.boot() }
            }
            .keyboardShortcut(.defaultAction)
        }
        .padding(32)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Palette.surface)
    }
}

private struct EmptyDetailView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: "bubble.left.and.text.bubble.right")
                .font(.system(size: 30))
                .foregroundStyle(Palette.tertiaryText)
            Text("No session selected")
                .font(Typography.title)
            Button("New Session") {
                Task { await model.newSession() }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Palette.surface)
    }
}
