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

    private var sessionsLoadedFor: Connection.ID?

    var selectedConnection: Connection? {
        connections.first { $0.id == selectedConnectionID }
    }

    var selectedSession: Session? {
        sessions.first { $0.id == selectedSessionID }
    }

    // MARK: - Startup

    /// Bring the local harness up and attach a connection to it.
    func boot() async {
        if connections.isEmpty {
            connections = [Connection(name: "This Mac", kind: .local, baseURL: nil)]
            selectedConnectionID = connections[0].id
        }
        await sidecar.start()
        guard case let .ready(port) = sidecar.state,
              let local = connections.first(where: { $0.kind == .local }) else {
            if case let .failed(reason) = sidecar.state { lastError = reason }
            return
        }
        local.baseURL = URL(string: "http://127.0.0.1:\(port)")
        local.isOnline = true
        await refresh()
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
                    return live
                }
                return Session(
                    id: summary.sessionID,
                    title: summary.tag ?? summary.preview ?? "New session",
                    client: client,
                    lastActivity: summary.updatedDate
                )
            }
            sessionsLoadedFor = connection.id

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
        for session in sessions { session.stop() }
        sidecar.stop()
    }
}
