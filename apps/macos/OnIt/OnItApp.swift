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
