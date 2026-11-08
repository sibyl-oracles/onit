import SwiftUI

/// Keys & Models / General / Connections / Diagnostics.
///
/// Keys & Models is first because it is the tab a first run needs: without a
/// model endpoint and its key the app cannot answer anything, and this is the
/// only place a secret is written from the app.
///
/// There is no permission-mode control here or in the composer: the backend
/// has no session flag to honour it yet (plan §11.5), and a menu that says
/// "Allow All" while the agent keeps asking is worse than no menu.
struct SettingsScene: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        TabView {
            KeysModelsTab()
                .tabItem { Label("Keys & Models", systemImage: "key.horizontal") }
            GeneralTab()
                .tabItem { Label("General", systemImage: "gearshape") }
            ConnectionsTab()
                .tabItem { Label("Connections", systemImage: "network") }
            DiagnosticsTab()
                .tabItem { Label("Diagnostics", systemImage: "stethoscope") }
        }
        .frame(width: 560, height: 420)
    }
}

private struct GeneralTab: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        Form {
            LabeledContent("Agent") {
                Text(model.config?.agent ?? "OnIt")
            }
            LabeledContent("Appearance") {
                Text("Follows the system")
            }
            LabeledContent("Text size") {
                Picker("Text size", selection: Binding(
                    get: { model.textScale },
                    set: { model.textScale = $0 }
                )) {
                    Text("Compact").tag(CGFloat(0.85))
                    Text("Default").tag(CGFloat(1.0))
                    Text("Large").tag(CGFloat(1.15))
                    Text("Largest").tag(CGFloat(1.3))
                }
                .labelsHidden()
                .frame(width: 120)
            }
        }
        .formStyle(.grouped)
        .padding()
    }
}

private struct ConnectionsTab: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        List {
            ForEach(model.connections) { connection in
                HStack {
                    Circle()
                        .fill(connection.status == .online ? Palette.success : Palette.danger)
                        .frame(width: 7, height: 7)
                    VStack(alignment: .leading, spacing: 1) {
                        Text(connection.name)
                            .font(Typography.body)
                        Text(connection.baseURL?.absoluteString ?? "starting…")
                            .font(Typography.caption)
                            .foregroundStyle(Palette.tertiaryText)
                    }
                    Spacer()
                    if connection.kind == .remote {
                        Button(role: .destructive) {
                            model.remove(connection)
                        } label: {
                            Image(systemName: "minus.circle")
                                .font(.system(size: 13))
                        }
                        .buttonStyle(.plain)
                        .help("Remove \(connection.name)")
                        .accessibilityLabel("Remove \(connection.name)")
                    }
                }
            }
        }
        .padding()
    }
}

private struct DiagnosticsTab: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(sidecarDescription)
                .font(Typography.caption)
                .foregroundStyle(Palette.secondaryText)

            ScrollView {
                Text(model.sidecar.logText)
                    .font(Typography.mono)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .padding(8)
            .background(Palette.raised, in: RoundedRectangle(cornerRadius: Metrics.controlRadius))
        }
        .padding()
    }

    private var sidecarDescription: String {
        switch model.sidecar.state {
        case .idle: return "Harness: idle"
        case .starting: return "Harness: starting…"
        case let .ready(port): return "Harness: ready on 127.0.0.1:\(port)"
        case let .failed(reason): return "Harness: failed — \(reason)"
        case .stopped: return "Harness: stopped"
        }
    }
}
