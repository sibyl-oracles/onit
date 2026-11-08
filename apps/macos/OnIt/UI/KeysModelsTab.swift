import SwiftUI

/// The Keys & Models tab: model endpoints and integration secrets.
///
/// The security property this tab is built on is the server's, not the app's:
/// **a key is never returned and never stored here.** Reads report where a
/// key comes from and at most its last four characters; writes send the key
/// once, on save, to the harness, which puts it in the OS keychain. The app
/// process never holds a secret after the field is cleared.
///
/// A remote connection with auth enabled refuses these writes (401). That is
/// shown as guidance — "sign in on this host's web UI" — rather than as a
/// broken form.
struct KeysModelsTab: View {
    @Environment(AppModel.self) private var model
    @State private var surface: OnItClient.EndpointSurface?
    @State private var loadError: String?
    @State private var isBusy = false
    @State private var editingEndpoint: EndpointDraft?
    @State private var keyPrompt: KeyPrompt?

    var body: some View {
        Group {
            if let surface {
                content(surface)
            } else if let loadError {
                VStack(spacing: 8) {
                    Text(loadError)
                        .font(Typography.body)
                        .foregroundStyle(Palette.secondaryText)
                    Button("Retry") { Task { await load() } }
                        .controlSize(.small)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ProgressView()
                    .controlSize(.small)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .padding()
        .task { await load() }
    }

    private func content(_ surface: OnItClient.EndpointSurface) -> some View {
        Form {
            if !surface.keychain {
                Text("No OS keychain is reachable; keys are stored in a 0600 secrets file next to the config.")
                    .font(Typography.caption)
                    .foregroundStyle(Palette.secondaryText)
            }

            Section("Model endpoints") {
                ForEach(surface.endpoints) { row in
                    EndpointRowView(row: row) {
                        editingEndpoint = EndpointDraft(from: row)
                    } onKey: {
                        keyPrompt = KeyPrompt(
                            title: row.displayTitle,
                            keyName: row.host,
                            isIntegration: false
                        )
                    } onForget: {
                        Task { await forgetEndpointKey(row.host) }
                    } onRemove: {
                        Task { await removeEndpoint(row) }
                    }
                }
                .onMove { indices, offset in
                    var rows = surface.endpoints
                    rows.move(fromOffsets: indices, toOffset: offset)
                    Task { await saveEndpoints(rows) }
                }

                Button {
                    editingEndpoint = EndpointDraft(presets: surface.presets)
                } label: {
                    Label("Add endpoint", systemImage: "plus")
                        .font(Typography.body)
                }
                .controlSize(.small)
            }

            Section("Integrations") {
                ForEach(surface.integrations) { row in
                    IntegrationRowView(row: row) {
                        keyPrompt = KeyPrompt(
                            title: row.displayName,
                            keyName: row.key,
                            isIntegration: true
                        )
                    } onForget: {
                        Task { await forgetIntegrationKey(row.key) }
                    }
                }
            }

            Section {
                LabeledContent("Keys are stored") {
                    Text(surface.keychain ? "In your macOS keychain, by the OnIt harness." : "In a 0600 secrets file, by the OnIt harness.")
                        .font(Typography.caption)
                }
                LabeledContent("Config") {
                    Text(surface.configPath ?? "")
                        .font(Typography.caption)
                        .textSelection(.enabled)
                }
            } footer: {
                Text("The app never sees a key after you save it. The harness reads it from the keychain when it calls the model.")
                    .font(Typography.caption)
            }
        }
        .formStyle(.grouped)
        .sheet(item: $editingEndpoint) { draft in
            EndpointEditSheet(draft: draft, presets: surface.presets) { result in
                Task { await apply(draft: draft, result: result, surface: surface) }
            }
        }
        .sheet(item: $keyPrompt) { prompt in
            KeyPromptSheet(prompt: prompt) { value in
                Task {
                    if prompt.isIntegration {
                        await storeIntegrationKey(prompt.keyName, value: value)
                    } else {
                        await storeEndpointKey(prompt.keyName, value: value)
                    }
                }
            }
        }
    }

    // MARK: - Actions

    private func load() async {
        guard let client = model.selectedConnection?.client else {
            loadError = "No connection."
            return
        }
        isBusy = true
        loadError = nil
        do {
            surface = try await client.endpointSurface()
        } catch {
            loadError = Self.message(error)
        }
        isBusy = false
    }

    private func reload() async {
        await load()
    }

    private func saveEndpoints(_ rows: [OnItClient.EndpointRow]) async {
        guard let client = model.selectedConnection?.client else { return }
        do {
            surface = try await client.saveEndpoints(rows)
        } catch {
            model.lastError = Self.message(error)
        }
    }

    private func apply(
        draft: EndpointDraft, result: EndpointDraft.Result,
        surface current: OnItClient.EndpointSurface
    ) async {
        guard let client = model.selectedConnection?.client else { return }
        do {
            // A key typed in the add form is stored before the endpoint is
            // saved: the add form asks for it first, and losing it at that
            // point would be the same bug as not asking.
            if let key = result.key, !key.isEmpty {
                _ = try await client.storeEndpointKey(host: result.host, key: key)
            }
            var rows = current.endpoints.filter { $0.host != draft.originalHost }
            let position = draft.insertIndex ?? rows.count
            let row = OnItClient.EndpointRow(
                id: result.host, host: result.host, model: result.model,
                name: result.name, priority: position, position: position,
                key: OnItClient.KeyStatus(source: nil, masked: nil, label: "none"),
                notes: []
            )
            rows.insert(row, at: min(position, rows.count))
            surface = try await client.saveEndpoints(rows)
        } catch {
            model.lastError = Self.message(error)
        }
    }

    private func removeEndpoint(_ row: OnItClient.EndpointRow) async {
        guard let client = model.selectedConnection?.client, let surface else { return }
        let rows = surface.endpoints.filter { $0.host != row.host }
        do {
            self.surface = try await client.saveEndpoints(rows)
        } catch {
            model.lastError = Self.message(error)
        }
    }

    private func storeEndpointKey(_ host: String, value: String) async {
        guard let client = model.selectedConnection?.client else { return }
        do {
            _ = try await client.storeEndpointKey(host: host, key: value)
            await reload()
        } catch {
            model.lastError = Self.message(error)
        }
    }

    private func forgetEndpointKey(_ host: String) async {
        guard let client = model.selectedConnection?.client else { return }
        do {
            _ = try await client.forgetEndpointKey(host: host)
            await reload()
        } catch {
            model.lastError = Self.message(error)
        }
    }

    private func storeIntegrationKey(_ keyName: String, value: String) async {
        guard let client = model.selectedConnection?.client else { return }
        do {
            _ = try await client.storeIntegrationKey(keyName: keyName, value: value)
            await reload()
        } catch {
            model.lastError = Self.message(error)
        }
    }

    private func forgetIntegrationKey(_ keyName: String) async {
        guard let client = model.selectedConnection?.client else { return }
        do {
            _ = try await client.forgetIntegrationKey(keyName: keyName)
            await reload()
        } catch {
            model.lastError = Self.message(error)
        }
    }

    static func message(_ error: Error) -> String {
        (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
    }
}

// MARK: - Rows

private struct EndpointRowView: View {
    let row: OnItClient.EndpointRow
    let onEdit: () -> Void
    let onKey: () -> Void
    let onForget: () -> Void
    let onRemove: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                VStack(alignment: .leading, spacing: 1) {
                    Text(row.displayTitle)
                        .font(Typography.bodyEmphasis)
                    Text(row.host)
                        .font(Typography.caption)
                        .foregroundStyle(Palette.secondaryText)
                        .textSelection(.enabled)
                }
                Spacer()
                Menu {
                    Button("Set API key…", action: onKey)
                    if row.key.isSet {
                        Button("Forget key", action: onForget)
                    }
                    Button("Edit…", action: onEdit)
                    Button("Remove", role: .destructive, action: onRemove)
                } label: {
                    Image(systemName: "ellipsis.circle")
                        .font(.system(size: 14))
                }
                .menuStyle(.borderlessButton)
                .fixedSize()
                .accessibilityLabel("Actions for \(row.displayTitle)")
            }

            HStack(spacing: 6) {
                Image(systemName: row.key.isSet ? "checkmark.shield" : "exclamationmark.shield")
                    .font(.system(size: 10))
                    .foregroundStyle(row.key.isSet ? Palette.success : Palette.danger)
                Text(row.key.label)
                    .font(Typography.caption)
                    .foregroundStyle(Palette.secondaryText)
                if let masked = row.key.masked {
                    Text(masked)
                        .font(Typography.caption.monospacedDigit())
                        .foregroundStyle(Palette.tertiaryText)
                }
            }

            if let model = row.model, !model.isEmpty {
                Text("Model: \(model)")
                    .font(Typography.caption)
                    .foregroundStyle(Palette.secondaryText)
            }

            ForEach(row.notes, id: \.self) { note in
                Label(note, systemImage: "info.circle")
                    .font(Typography.caption)
                    .foregroundStyle(Color.orange)
            }
        }
        .padding(.vertical, 2)
    }
}

private struct IntegrationRowView: View {
    let row: OnItClient.IntegrationRow
    let onSet: () -> Void
    let onForget: () -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            VStack(alignment: .leading, spacing: 2) {
                Text(row.displayName)
                    .font(Typography.bodyEmphasis)
                Text(row.blurb)
                    .font(Typography.caption)
                    .foregroundStyle(Palette.secondaryText)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 4) {
                HStack(spacing: 4) {
                    Image(systemName: row.isSet ? "checkmark.shield" : "shield")
                        .font(.system(size: 10))
                        .foregroundStyle(row.isSet ? Palette.success : Palette.tertiaryText)
                    Text(row.status)
                        .font(Typography.caption)
                        .foregroundStyle(Palette.secondaryText)
                    if let masked = row.masked {
                        Text(masked)
                            .font(Typography.caption.monospacedDigit())
                            .foregroundStyle(Palette.tertiaryText)
                    }
                }
                HStack(spacing: 6) {
                    Button("Set…", action: onSet)
                        .controlSize(.small)
                    if row.isSet && row.source == "keychain" {
                        Button("Forget", action: onForget)
                            .controlSize(.small)
                    }
                }
            }
        }
        .padding(.vertical, 2)
    }
}

// MARK: - Sheets

/// What the add/edit form holds while it is open.
struct EndpointDraft: Identifiable {
    struct Preset: Identifiable {
        let label: String
        let host: String
        let needsKey: Bool
        var id: String { host }
    }

    let originalHost: String?
    /// Where a new endpoint goes in the preference order.
    let insertIndex: Int?
    let presets: [Preset]

    var id: String { originalHost ?? "new" }

    init(from row: OnItClient.EndpointRow) {
        self.originalHost = row.host
        self.insertIndex = row.position
        self.presets = []
    }

    init(presets: [OnItClient.EndpointPreset]) {
        self.originalHost = nil
        self.insertIndex = nil
        self.presets = presets.map { Preset(label: $0.label, host: $0.host, needsKey: $0.needsKey) }
    }
}

/// What the add/edit form hands back.
extension EndpointDraft {
    struct Result {
        let host: String
        let model: String?
        let name: String?
        /// Typed in this form, to be stored before the endpoint is saved.
        let key: String?
    }
}

struct EndpointEditSheet: View {
    let draft: EndpointDraft
    let presets: [OnItClient.EndpointPreset]
    let onDone: (EndpointDraft.Result) -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var host: String = ""
    @State private var modelName: String = ""
    @State private var name: String = ""
    @State private var key: String = ""
    @State private var needsKey = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(draft.originalHost == nil ? "Add endpoint" : "Edit endpoint")
                .font(Typography.title)

            if draft.originalHost == nil, !presets.isEmpty {
                Picker("Preset", selection: $host) {
                    Text("Custom…").tag("")
                    ForEach(presets) { preset in
                        Text(preset.label).tag(preset.host)
                    }
                }
                .onChange(of: host) {
                    needsKey = presets.first { $0.host == host }?.needsKey ?? false
                }
            }

            TextField("Endpoint URL", text: $host)
                .textFieldStyle(.roundedBorder)
            TextField("Model (e.g. llama4, openai/gpt-5)", text: $modelName)
                .textFieldStyle(.roundedBorder)
            TextField("Name (optional)", text: $name)
                .textFieldStyle(.roundedBorder)
            SecureField(needsKey ? "API key" : "API key (if needed)", text: $key)
                .textFieldStyle(.roundedBorder)

            Text("The key goes to your keychain, not to the config file, and is never sent anywhere but this host.")
                .font(Typography.caption)
                .foregroundStyle(Palette.secondaryText)

            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Button(draft.originalHost == nil ? "Add" : "Save") { save() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(host.trimmingCharacters(in: .whitespaces).isEmpty)
            }
        }
        .padding(20)
        .frame(width: 420)
        .onAppear {
            if let original = draft.originalHost {
                host = original
            }
        }
    }

    private func save() {
        let trimmed = host.trimmingCharacters(in: .whitespacesAndNewlines)
        var raw = trimmed
        if !raw.contains("://") { raw = "http://" + raw }
        onDone(
            EndpointDraft.Result(
                host: raw,
                model: modelName.isEmpty ? nil : modelName,
                name: name.isEmpty ? nil : name,
                key: key.isEmpty ? nil : key
            )
        )
        dismiss()
    }
}

/// The ask-a-key sheet. The value lives only in this field until Save.
struct KeyPrompt: Identifiable {
    let title: String
    let keyName: String
    let isIntegration: Bool
    var id: String { keyName }
}

struct KeyPromptSheet: View {
    let prompt: KeyPrompt
    let onDone: (String) -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var value = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("\(prompt.title) API key")
                .font(Typography.title)
            SecureField("Paste the key", text: $value)
                .textFieldStyle(.roundedBorder)
            Text("Stored in your keychain by the OnIt harness. The app keeps no copy.")
                .font(Typography.caption)
                .foregroundStyle(Palette.secondaryText)
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Button("Save") {
                    onDone(value)
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
                .disabled(value.trimmingCharacters(in: .whitespaces).isEmpty)
            }
        }
        .padding(20)
        .frame(width: 400)
    }
}

// MARK: - Row display helpers

extension OnItClient.EndpointRow {
    /// The list shows the saved name, else the model, else the host.
    var displayTitle: String {
        if let name, !name.isEmpty { return name }
        if let model, !model.isEmpty { return model }
        return host
    }
}
