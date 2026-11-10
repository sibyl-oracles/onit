import AppKit
import SwiftUI

@main
struct OnItApp: App {
    @State private var model = AppModel()
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(model)
                .environment(model.audio)
                .environment(model.video)
                .environment(model.dictation)
                .environment(model.tts)
                // Adopt the persisted scale before the first layout pass so
                // the transcript never draws once at the wrong size.
                .onAppear { model.adoptPersistedTextScale() }
                .frame(minWidth: 720, minHeight: 480)
                .task { await model.boot() }
                // Coming back to the app is the moment stale state is most
                // visible: a host that went away and came back, or a harness
                // that died. One probe now, rather than a wrong dot for ten.
                .onChange(of: scenePhase) { _, phase in
                    // Backgrounded windows stop the 10 s poll; the probe on
                    // the way back to active is what heals stale state.
                    model.isHealthPollPaused = phase != .active
                    if phase == .active {
                        Task { await model.checkHealth() }
                    }
                }
                // Quitting must reap the sidecar harness. Without this the
                // `python -m onit.cli serve web` child outlives the app: every
                // relaunch allocates a fresh port and spawns a new harness,
                // and the orphans accumulate — 43 were found on one machine.
                // scenePhase does not fire on quit, so this observer is the
                // only reliable teardown hook in a SwiftUI app lifecycle.
                .onReceive(NotificationCenter.default.publisher(
                    for: NSApplication.willTerminateNotification
                )) { _ in
                    model.shutdown()
                }
        }
        .windowToolbarStyle(.unified(showsTitle: false))
        .commands {
            CommandGroup(after: .newItem) {
                Button("New Session") {
                    Task { await model.newSession() }
                }
                .keyboardShortcut("n", modifiers: .command)
            }
            CommandMenu("Session") {
                Button("Stop") {
                    model.selectedSession?.stop()
                }
                .keyboardShortcut(".", modifiers: .command)
                .disabled(!(model.selectedSession?.isProcessing ?? false))

                Button("Refresh") {
                    Task { await model.refresh() }
                }
                .keyboardShortcut("r", modifiers: .command)
            }
        }

        Settings {
            SettingsScene()
                .environment(model)
        }
    }
}
