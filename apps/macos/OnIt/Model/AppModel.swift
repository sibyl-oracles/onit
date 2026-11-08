import Foundation
import Observation

/// The app's single source of truth: one sidecar, one or more connections,
/// and the sessions belonging to whichever connection is selected.
@MainActor
@Observable
final class AppModel {
    let sidecar = SidecarManager()

    var connections: [Connection] = []
    var selectedConnectionID: Connection.ID?
    var sessions: [Session] = []
    var selectedSessionID: Session.ID?
    var config: OnItClient.Config?
    var lastError: String?

    /// The user's text size, as a multiplier on Typography's design sizes.
    /// Its didSet is the only writer of `Typography.scale`: the type-scale
    /// enum is plain statics so equality-gated views can read it without an
    /// environment hop, and the @Observable graph is what propagates change.
    var textScale: CGFloat = {
        let stored = UserDefaults.standard.double(forKey: "onit.textScale")
        return stored > 0 ? CGFloat(stored) : 1.0
    }() {
        didSet {
            Typography.scale = textScale
            UserDefaults.standard.set(Double(textScale), forKey: "onit.textScale")
        }
    }

    /// Inline audio playback, shared across sessions: one sound at a time, so
    /// starting a row in one transcript stops the row playing in another.
    let audio = AudioCenter()

    /// Sync Typography with whatever a previous launch persisted. Property
    /// initializers cannot touch other statics safely at declaration time.
    func adoptPersistedTextScale() {
        Typography.scale = textScale
    }

    /// Download an attachment's bytes for inline playback. Public so file
    /// rows can reach it through the environment without knowing which
    /// connection owns the URL.
    func attachmentData(_ path: String) async -> Data? {
        guard let client = selectedConnection?.client else { return nil }
        return try? await client.data(path)
    }

    var selectedConnection: Connection? {
        connections.first { $0.id == selectedConnectionID }
    }

    var selectedSession: Session? {
        sessions.first { $0.id == selectedSessionID }
    }

    // MARK: - Startup

    /// Bring the local harness up and attach a connection to it.
    func boot() async {
        if connections.isEmpty { loadPersistedConnections() }
        if connections.isEmpty {
            connections = [Connection(name: "This Mac", kind: .local, baseURL: nil)]
        }
        if selectedConnectionID == nil || !connections.contains(where: { $0.id == selectedConnectionID }) {
            selectedConnectionID = connections.first?.id
        }
        sidecar.onDeath = { [weak self] in self?.localConnectionDied() }
        await sidecar.start()
        guard case let .ready(port) = sidecar.state,
              let local = connections.first(where: { $0.kind == .local }) else {
            if case let .failed(reason) = sidecar.state { lastError = reason }
            return
        }
        local.baseURL = URL(string: "http://127.0.0.1:\(port)")
        local.isOnline = true
        await refresh()
        startHealthMonitor()
    }

    // MARK: - Sessions

    func refresh() async {
        guard let connection = selectedConnection,
              let client = connection.client else { return }

        do {
            config = try await client.config()
            let summaries = try await client.sessions()

            // Keep live view-models for sessions that still exist, so a
            // refresh does not drop a stream that is mid-flight.
            let existing = Dictionary(uniqueKeysWithValues: sessions.map { ($0.id, $0) })
            sessions = summaries.map { summary in
                if let live = existing[summary.sessionID] {
                    if summary.tag != nil { live.title = summary.tag! }
                    live.lastActivity = summary.updatedDate
                    live.summaryPreview = summary.preview
                    return live
                }
                return Session(
                    id: summary.sessionID,
                    title: summary.tag ?? summary.preview ?? "New session",
                    client: client,
                    lastActivity: summary.updatedDate,
                    summaryPreview: summary.preview
                )
            }

            if selectedSessionID == nil || !sessions.contains(where: { $0.id == selectedSessionID }) {
                selectedSessionID = sessions.first?.id
            }
            if let selected = selectedSession {
                await selected.load()
            }
        } catch {
            lastError = error.localizedDescription
            connection.isOnline = false
        }
    }

    func select(_ connection: Connection) async {
        guard connection.id != selectedConnectionID else { return }
        selectedConnectionID = connection.id
        sessions = []
        selectedSessionID = nil
        await refresh()
    }

    func newSession() async {
        guard let client = selectedConnection?.client else { return }
        do {
            let id = try await client.newSession()
            let session = Session(id: id, title: "New session", client: client)
            session.isLoaded = true
            sessions.insert(session, at: 0)
            selectedSessionID = id
        } catch {
            lastError = error.localizedDescription
        }
    }

    func delete(_ session: Session) async {
        do {
            try await session.client.delete(session: session.id)
            sessions.removeAll { $0.id == session.id }
            if selectedSessionID == session.id {
                selectedSessionID = sessions.first?.id
                if let next = selectedSession { await next.load() }
            }
        } catch {
            lastError = error.localizedDescription
        }
    }

    func rename(_ session: Session, to tag: String) async {
        let trimmed = tag.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        do {
            try await session.client.rename(session: session.id, tag: trimmed)
            session.title = trimmed
        } catch {
            lastError = error.localizedDescription
        }
    }

    // MARK: - Grouping

    /// Sessions bucketed the way a person remembers them.
    struct Group: Identifiable {
        let id: String
        let sessions: [Session]
    }

    func grouped(_ sessions: [Session]) -> [Group] {
        let calendar = Calendar.current
        let now = Date()
        var buckets: [(String, [Session])] = [
            ("Today", []), ("Yesterday", []), ("Previous 7 days", []), ("Older", []),
        ]
        for session in sessions {
            // A session created a moment ago has no server timestamp yet; it
            // belongs at the top, not in "Older".
            let date = session.lastActivity ?? now
            let index: Int
            if calendar.isDateInToday(date) {
                index = 0
            } else if calendar.isDateInYesterday(date) {
                index = 1
            } else if let week = calendar.date(byAdding: .day, value: -7, to: now), date > week {
                index = 2
            } else {
                index = 3
            }
            buckets[index].1.append(session)
        }
        return buckets
            .filter { !$0.1.isEmpty }
            .map { Group(id: $0.0, sessions: $0.1) }
    }

    // MARK: - Teardown

    func shutdown() {
        healthTask?.cancel()
        healthTask = nil
        for session in sessions { session.stop() }
        sidecar.stop()
    }

    // MARK: - Connection persistence

    /// Remote connections survive a relaunch. The local one is not persisted:
    /// it is this app's own harness, not a user choice, and its port changes
    /// every launch anyway.
    private static let connectionsKey = "onit.connections.v1"

    private func loadPersistedConnections() {
        guard let data = UserDefaults.standard.data(forKey: Self.connectionsKey),
              let snapshots = try? JSONDecoder().decode([Connection.Snapshot].self, from: data)
        else { return }
        let remote = snapshots.filter { $0.kind == .remote }.map { $0.connection() }
        guard !remote.isEmpty else { return }
        connections = [Connection(name: "This Mac", kind: .local, baseURL: nil)] + remote
    }

    private func persistConnections() {
        let snapshots = connections.filter { $0.kind == .remote }.map(Connection.Snapshot.init(from:))
        guard let data = try? JSONEncoder().encode(snapshots) else { return }
        UserDefaults.standard.set(data, forKey: Self.connectionsKey)
    }

    func addRemoteConnection(name: String, baseURL: URL) {
        let connection = Connection(name: name, kind: .remote, baseURL: baseURL)
        connections.append(connection)
        persistConnections()
    }

    /// Remove a connection. The local one is load-bearing — it is where the
    /// sidecar attaches — so it is not removable.
    func remove(_ connection: Connection) {
        guard connection.kind == .remote else { return }
        connections.removeAll { $0.id == connection.id }
        persistConnections()
        if selectedConnectionID == connection.id {
            selectedConnectionID = connections.first?.id
            sessions = []
            selectedSessionID = nil
            Task { await refresh() }
        }
    }

    // MARK: - Harness death

    /// How many times this run has restarted the harness on its own. A cap
    /// turns a crash loop into a visible failure instead of a silent spin;
    /// "Try Again" in the failure view still works regardless.
    private var restartAttempts = 0

    private func localConnectionDied() {
        guard let local = connections.first(where: { $0.kind == .local }) else { return }
        local.isOnline = false
        guard restartAttempts < 3 else {
            lastError = "The OnIt harness exited unexpectedly and could not be restarted. See Settings → Diagnostics for its log."
            return
        }
        restartAttempts += 1
        Task { [weak self] in
            try? await Task.sleep(for: .seconds(1))
            await self?.boot()
        }
    }

    // MARK: - Health

    private var healthTask: Task<Void, Never>?
    /// False while the window is backgrounded: the poll is pointless when
    /// nobody is looking, and the activation probe (scenePhase) covers the
    /// wake-up case.
    var isHealthPollPaused = false

    /// Reconcile the online dots with reality, every ten seconds.
    ///
    /// A connection can go stale in both directions: a remote host that came
    /// back up, or the local harness that died while another window had
    /// focus. The poll is what heals both without a relaunch.
    func startHealthMonitor() {
        guard healthTask == nil else { return }
        healthTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(10))
                guard let self, !self.isHealthPollPaused else { continue }
                await self.checkHealth()
            }
        }
    }

    /// Ping every connection that has an address. Parallel: five offline
    /// hosts at three seconds each must not serialize into a fifteen-second
    /// stall on the main actor.
    func checkHealth() async {
        let clients = connections.compactMap { connection -> (Connection, OnItClient)? in
            connection.client.map { (connection, $0) }
        }
        await withTaskGroup(of: (Connection.ID, Bool).self) { group in
            for (connection, client) in clients {
                group.addTask { (connection.id, await client.ping()) }
            }
            for await (id, alive) in group {
                if let connection = connections.first(where: { $0.id == id }) {
                    connection.isOnline = alive
                }
            }
        }
    }
}
