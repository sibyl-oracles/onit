import SwiftUI

/// The narrow column of OnIts.
///
/// One row today — "This Mac" — because a rail with a single item looks like a
/// bug unless it is drawn as the place a second one will go. The add button
/// opens the remote-host sheet, which is the same `OnItClient` pointed at a
/// different base URL.
struct ConnectionRail: View {
    @Environment(AppModel.self) private var model
    @State private var showingAddHost = false

    var body: some View {
        VStack(spacing: 10) {
            ForEach(model.connections) { connection in
                Button {
                    Task { await model.select(connection) }
                } label: {
                    ConnectionBadge(
                        connection: connection,
                        isSelected: connection.id == model.selectedConnectionID
                    )
                }
                .buttonStyle(.plain)
                .help(connection.name)
                .accessibilityLabel("Connection \(connection.name)")
                .contextMenu {
                    if connection.kind == .remote {
                        Button("Remove “\(connection.name)”…", role: .destructive) {
                            model.remove(connection)
                        }
                    }
                }
            }

            Button {
                showingAddHost = true
            } label: {
                Image(systemName: "plus")
                    .font(.system(size: 13, weight: .medium))
                    .frame(width: 36, height: 36)
                    .background(Palette.raised, in: RoundedRectangle(cornerRadius: 10))
                    .foregroundStyle(Palette.secondaryText)
            }
            .buttonStyle(.plain)
            .help("Connect to a host")
            .accessibilityLabel("Connect to a host")

            Spacer()
        }
        .padding(.top, 12)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Palette.surface)
        .sheet(isPresented: $showingAddHost) {
            AddHostSheet()
                .environment(model)
        }
    }
}

private struct ConnectionBadge: View {
    let connection: Connection
    let isSelected: Bool

    var body: some View {
        ZStack(alignment: .bottomTrailing) {
            RoundedRectangle(cornerRadius: 10)
                .fill(isSelected ? Color.accentColor.opacity(0.16) : Palette.raised)
                .frame(width: 36, height: 36)
                .overlay {
                    Text(initials)
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(isSelected ? Color.accentColor : Palette.secondaryText)
                }

            Circle()
                .fill(dotColor)
                .frame(width: 8, height: 8)
                .overlay(Circle().stroke(Palette.surface, lineWidth: 1.5))
                .offset(x: 2, y: 2)
        }
    }

    private var initials: String {
        connection.kind == .local ? "⌘" : String(connection.name.prefix(1)).uppercased()
    }

    private var dotColor: Color {
        switch connection.status {
        case .online: return Palette.success
        case .offline: return Palette.danger
        case .starting: return Color.orange
        }
    }
}

/// Point the app at an OnIt running somewhere else.
///
/// This is the same contract over a different base URL — a `serve web` on a
/// workstation, reached over the LAN. Nothing about the session model changes.
private struct AddHostSheet: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss

    @State private var name = ""
    @State private var address = ""
    @State private var error: String?
    @State private var isChecking = false

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Connect to a host")
                .font(Typography.title)

            Text("An OnIt running with `onit serve web` on another machine.")
                .font(Typography.caption)
                .foregroundStyle(Palette.secondaryText)

            TextField("Name", text: $name, prompt: Text("Workstation"))
                .textFieldStyle(.roundedBorder)
            TextField("Address", text: $address, prompt: Text("http://192.168.1.20:9000"))
                .textFieldStyle(.roundedBorder)

            if let error {
                Text(error)
                    .font(Typography.caption)
                    .foregroundStyle(Palette.danger)
            }

            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Button(isChecking ? "Checking…" : "Connect") { connect() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(address.isEmpty || isChecking)
            }
        }
        .padding(20)
        .frame(width: 380)
    }

    private func connect() {
        var raw = address.trimmingCharacters(in: .whitespacesAndNewlines)
        if !raw.contains("://") { raw = "http://" + raw }
        guard let url = URL(string: raw), url.host != nil else {
            error = "That does not look like an address."
            return
        }
        isChecking = true
        error = nil

        Task {
            let client = OnItClient(baseURL: url)
            do {
                _ = try await client.config()
                let connection = Connection(
                    name: name.isEmpty ? (url.host ?? "Remote") : name,
                    kind: .remote,
                    baseURL: url
                )
                connection.isOnline = true
                model.addRemoteConnection(name: connection.name, baseURL: url)
                if let added = model.connections.last {
                    await model.select(added)
                }
                dismiss()
            } catch {
                self.error = "No OnIt answered at \(url.absoluteString)."
            }
            isChecking = false
        }
    }
}
