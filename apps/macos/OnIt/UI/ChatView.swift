import SwiftUI

/// The transcript and the composer for one session.
struct ChatView: View {
    @Bindable var session: Session
    @State private var draft = ""
    @State private var attachments: [URL] = []
    @State private var isUploading = false
    @State private var uploadError: String?
    @State private var isDropTargeted = false
    @FocusState private var composerFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            transcript
            Divider().overlay(Palette.hairline)
            composer
        }
        .background(Palette.surface)
        .navigationTitle(session.title)
        .task { await session.load() }
        .onAppear { composerFocused = true }
    }

    // MARK: - Transcript

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 18) {
                    if session.messages.isEmpty {
                        EmptyTranscript()
                            .padding(.top, 80)
                    }
                    ForEach(session.messages) { message in
                        MessageBubble(message: message)
                            .id(message.id)
                    }
                    // The approval card sits in the flow, next to the turn it
                    // gates, so the command it names is still on screen. The
                    // card gets the absolute deadline and ticks itself.
                    if let approval = session.pendingApproval {
                        ApprovalCard(
                            approval: approval,
                            deadline: session.approvalDeadline
                        ) { decision in
                            session.answer(approval, decision: decision)
                        }
                        .id("approval")
                    }
                    Color.clear.frame(height: 1).id("bottom")
                }
                .padding(.horizontal, Metrics.gutter)
                .padding(.vertical, 20)
                .frame(maxWidth: Metrics.transcriptMaxWidth, alignment: .leading)
                .frame(maxWidth: .infinity)
            }
            .onChange(of: session.messages.count) {
                withAnimation(.onit) { proxy.scrollTo("bottom", anchor: .bottom) }
            }
            .onChange(of: session.messages.last?.content) {
                proxy.scrollTo("bottom", anchor: .bottom)
            }
            .onChange(of: session.pendingApproval?.id) { _, newValue in
                if newValue != nil {
                    withAnimation(.onit) { proxy.scrollTo("approval", anchor: .center) }
                    // A gated command must never expire because the window was
                    // behind another one.
                    NSApp.activate(ignoringOtherApps: true)
                }
            }
        }
    }

    // MARK: - Composer

    private var composer: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !attachments.isEmpty {
                HStack(spacing: 6) {
                    ForEach(attachments, id: \.self) { url in
                        HStack(spacing: 4) {
                            Image(systemName: "paperclip")
                                .font(.system(size: 10))
                            Text(url.lastPathComponent)
                                .font(Typography.caption)
                                .lineLimit(1)
                            Button {
                                attachments.removeAll { $0 == url }
                            } label: {
                                Image(systemName: "xmark")
                                    .font(.system(size: 8))
                            }
                            .buttonStyle(.plain)
                            .accessibilityLabel("Remove \(url.lastPathComponent)")
                        }
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Palette.raised, in: Capsule())
                    }
                }
            }

            if isUploading {
                HStack(spacing: 6) {
                    ProgressView()
                        .controlSize(.mini)
                    Text("Uploading…")
                        .font(Typography.caption)
                        .foregroundStyle(Palette.secondaryText)
                }
            }

            if let uploadError {
                Text(uploadError)
                    .font(Typography.caption)
                    .foregroundStyle(Palette.danger)
                    .textSelection(.enabled)
            }

            HStack(alignment: .bottom, spacing: 8) {
                Button(action: attach) {
                    Image(systemName: "paperclip")
                        .font(.system(size: 13))
                        .foregroundStyle(Palette.secondaryText)
                }
                .buttonStyle(.plain)
                .disabled(session.isProcessing || isUploading)
                .help("Attach a file")
                .accessibilityLabel("Attach a file")

                TextField(
                    "Ask OnIt…",
                    text: $draft,
                    axis: .vertical
                )
                .textFieldStyle(.plain)
                .font(Typography.body)
                .lineLimit(1...8)
                .focused($composerFocused)
                .onSubmit(send)
                // An upload in flight means the send this field could trigger
                // would race the attachments it is about to carry.
                .disabled(session.isProcessing || isUploading)

                if session.isProcessing {
                    Button(action: session.stop) {
                        Image(systemName: "stop.circle.fill")
                            .font(.system(size: 18))
                            .foregroundStyle(Palette.secondaryText)
                    }
                    .buttonStyle(.plain)
                    .help("Stop (⌘.)")
                    .accessibilityLabel("Stop generating")
                } else {
                    Button(action: send) {
                        Image(systemName: "arrow.up.circle.fill")
                            .font(.system(size: 18))
                            .foregroundStyle(
                                canSend ? Color.accentColor : Palette.tertiaryText
                            )
                    }
                    .buttonStyle(.plain)
                    .disabled(!canSend)
                    .keyboardShortcut(.return, modifiers: .command)
                    .accessibilityLabel("Send message")
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .background(Palette.raised, in: RoundedRectangle(cornerRadius: Metrics.cardRadius))
            .overlay {
                RoundedRectangle(cornerRadius: Metrics.cardRadius)
                    .stroke(Palette.hairline, lineWidth: 1)
            }
            .overlay {
                // Where a dropped file would land. Only shown while a drag is
                // over the composer, so it costs nothing at rest.
                if isDropTargeted {
                    RoundedRectangle(cornerRadius: Metrics.cardRadius)
                        .stroke(Color.accentColor, lineWidth: 2)
                        .background(Color.accentColor.opacity(0.08), in: RoundedRectangle(cornerRadius: Metrics.cardRadius))
                }
            }
            .dropDestination(for: URL.self) { urls, _ in
                let files = urls.filter { $0.isFileURL }
                guard !files.isEmpty else { return false }
                attachments.append(contentsOf: files)
                composerFocused = true
                return true
            } isTargeted: { hovering in
                isDropTargeted = hovering
            }
        }
        .padding(.horizontal, Metrics.gutter)
        .padding(.vertical, 12)
        .frame(maxWidth: Metrics.transcriptMaxWidth)
        .frame(maxWidth: .infinity)
    }

    private var canSend: Bool {
        !session.isProcessing
            && !isUploading
            && !(draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && attachments.isEmpty)
    }

    private func send() {
        guard canSend else { return }
        let text = draft
        let files = attachments
        draft = ""
        attachments = []
        uploadError = nil
        Task {
            var paths: [String] = []
            var failures: [String] = []
            if !files.isEmpty {
                isUploading = true
                for url in files {
                    do {
                        paths.append(try await session.client.upload(session: session.id, fileURL: url))
                    } catch {
                        failures.append("\(url.lastPathComponent) — \(error.localizedDescription)")
                    }
                }
                isUploading = false
            }
            // A failed upload must not vanish: the user believes the model got
            // the file. Restore the draft and the chips, name what failed, and
            // send nothing — a partial attachment set is a silent wrong answer.
            if !failures.isEmpty {
                draft = text
                attachments = files
                uploadError = "Not uploaded:\n" + failures.joined(separator: "\n")
                return
            }
            session.send(text, files: paths)
        }
    }

    private func attach() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.message = "Choose files to send with your message"
        if panel.runModal() == .OK {
            attachments.append(contentsOf: panel.urls)
        }
    }
}

private struct EmptyTranscript: View {
    var body: some View {
        VStack(spacing: 10) {
            Text("What should OnIt do?")
                .font(.system(size: 17, weight: .medium))
            Text("It can read files, run commands and browse — and it will ask before anything destructive.")
                .font(Typography.body)
                .foregroundStyle(Palette.secondaryText)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 380)
        }
        .frame(maxWidth: .infinity)
    }
}
