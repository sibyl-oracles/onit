import XCTest
@testable import OnIt

/// Recency grouping and the connection model — the pure parts of AppModel.
@MainActor
final class AppModelTests: XCTestCase {
    private func makeSession(lastActivity: Date?) -> Session {
        Session(
            id: UUID().uuidString,
            title: "s",
            client: OnItClient(baseURL: URL(string: "http://127.0.0.1:9")!),
            lastActivity: lastActivity
        )
    }

    func testGroupedBuckets() {
        let model = AppModel()
        let calendar = Calendar.current
        let now = Date()

        let today = makeSession(lastActivity: now)
        let yesterday = makeSession(lastActivity: calendar.date(byAdding: .day, value: -1, to: now))
        let thisWeek = makeSession(lastActivity: calendar.date(byAdding: .day, value: -3, to: now))
        let old = makeSession(lastActivity: calendar.date(byAdding: .day, value: -30, to: now))

        let groups = model.grouped([old, today, thisWeek, yesterday])
        let names = groups.map(\.id)
        // Buckets appear in order, and only when non-empty.
        XCTAssertEqual(names, ["Today", "Yesterday", "Previous 7 days", "Older"])
        XCTAssertEqual(groups[0].sessions, [today])
        XCTAssertEqual(groups[1].sessions, [yesterday])
        XCTAssertEqual(groups[2].sessions, [thisWeek])
        XCTAssertEqual(groups[3].sessions, [old])
    }

    func testSessionWithoutTimestampBelongsAtTop() {
        let model = AppModel()
        let fresh = makeSession(lastActivity: nil)
        let old = makeSession(lastActivity: Date().addingTimeInterval(-86_400 * 40))

        let groups = model.grouped([old, fresh])
        XCTAssertEqual(groups.first?.id, "Today", "a session created a moment ago has no timestamp yet")
        XCTAssertEqual(groups.first?.sessions, [fresh])
    }

    func testEmptyInputYieldsNoGroups() {
        let model = AppModel()
        XCTAssertTrue(model.grouped([]).isEmpty)
    }

    // MARK: - Connection persistence

    func testSnapshotRoundTrip() throws {
        let connection = Connection(name: "Workstation", kind: .remote, baseURL: URL(string: "http://10.0.0.5:9000"))
        let snapshot = Connection.Snapshot(from: connection)
        let data = try JSONEncoder().encode([snapshot])
        let decoded = try JSONDecoder().decode([Connection.Snapshot].self, from: data)

        XCTAssertEqual(decoded, [snapshot])
        let restored = try XCTUnwrap(decoded.first?.connection())
        XCTAssertEqual(restored.name, "Workstation")
        XCTAssertEqual(restored.kind, .remote)
        XCTAssertEqual(restored.baseURL, URL(string: "http://10.0.0.5:9000"))
        // State is deliberately not carried across a relaunch.
        XCTAssertFalse(restored.isOnline)
    }

    func testLocalConnectionIsNotRemovable() {
        let model = AppModel()
        model.connections = [Connection(name: "This Mac", kind: .local, baseURL: nil)]
        model.selectedConnectionID = model.connections[0].id

        model.remove(model.connections[0])

        XCTAssertEqual(model.connections.count, 1, "the local connection is load-bearing")
    }

    func testRemovingSelectedConnectionFallsBackToLocal() {
        let model = AppModel()
        let local = Connection(name: "This Mac", kind: .local, baseURL: nil)
        let remote = Connection(name: "Lab", kind: .remote, baseURL: URL(string: "http://10.0.0.9:9000"))
        model.connections = [local, remote]
        model.selectedConnectionID = remote.id

        model.remove(remote)

        XCTAssertEqual(model.connections, [local])
        XCTAssertEqual(model.selectedConnectionID, local.id)
    }

    func testConnectionStatus() {
        let local = Connection(name: "This Mac", kind: .local, baseURL: nil)
        XCTAssertEqual(local.status, .starting, "the local dot is orange until the sidecar reports")

        let remote = Connection(name: "Lab", kind: .remote, baseURL: URL(string: "http://10.0.0.9:9000"))
        XCTAssertEqual(remote.status, .offline)
        remote.isOnline = true
        XCTAssertEqual(remote.status, .online)
    }
}

@MainActor
final class TextScaleTests: XCTestCase {
    func testScalePropagatesToTypography() {
        let model = AppModel()
        defer { model.textScale = 1.0 }
        model.textScale = 1.3
        XCTAssertEqual(Typography.scale, 1.3)
        XCTAssertEqual(Typography.lineSpacing, 5.5 * 1.3)
    }

    func testAdoptPersistedScaleSyncsBeforeFirstLayout() {
        // The launch path: a stored value must reach Typography without
        // waiting for the picker's didSet to run once.
        Typography.scale = 1.0
        let model = AppModel()
        model.textScale = 0.85
        model.adoptPersistedTextScale()
        XCTAssertEqual(Typography.scale, 0.85)
        model.textScale = 1.0
    }
}
