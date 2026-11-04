import SwiftUI

@main
struct OnItApp: App {
    @State private var model = AppModel()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(model)
                .frame(minWidth: 860, minHeight: 520)
                .task { await model.boot() }
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
