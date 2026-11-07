import Foundation
import Observation

/// One OnIt the app is talking to: the local sidecar, or a remote host.
///
/// The rail exists so this distinction is a *row*, not a mode. A session on
/// the local sidecar and a session on a remote host are two `Session`
/// view-models against two `OnItClient`s; nothing downstream can tell them
/// apart, which is what makes adding a second connection cheap.
@MainActor
@Observable
final class Connection: Identifiable, Equatable {
    /// Identity equality — a Connection is its row in the rail.
    nonisolated static func == (lhs: Connection, rhs: Connection) -> Bool { lhs === rhs }

    /// String-backed so it can round-trip through the persisted snapshot.
    enum Kind: String, Codable, Sendable, Equatable {
        case local
        case remote
    }

    let id = UUID()
    var name: String
    let kind: Kind
    /// Nil for the local connection until the sidecar reports its port.
    var baseURL: URL?
    var isOnline = false

    init(name: String, kind: Kind, baseURL: URL?) {
        self.name = name
        self.kind = kind
        self.baseURL = baseURL
    }

    var client: OnItClient? {
        baseURL.map(OnItClient.init)
    }

    /// The status dot in the rail.
    enum Status: Equatable {
        case online, offline, starting
    }

    var status: Status {
        if isOnline { return .online }
        if kind == .local { return .starting }
        return .offline
    }

    /// What gets written to `UserDefaults`. Deliberately narrow — identity is
    /// rebuilt on load, so a relaunch gets fresh UUIDs and fresh state. The
    /// struct itself is unisolated so Codable can synthesize; only the two
    /// methods that touch a `Connection` need the main actor.
    struct Snapshot: Codable, Equatable {
        var name: String
        var kind: Kind
        var baseURL: URL?

        @MainActor
        init(from connection: Connection) {
            self.name = connection.name
            self.kind = connection.kind
            self.baseURL = connection.baseURL
        }

        @MainActor
        func connection() -> Connection {
            Connection(name: name, kind: kind, baseURL: baseURL)
        }
    }
}
