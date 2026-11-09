import XCTest
@testable import OnIt

@MainActor
final class VideoCenterTests: XCTestCase {
    override func tearDown() {
        VideoCenter().stopAll()
        super.tearDown()
    }

    func testPlayLoadsAndBecomesActive() {
        let center = VideoCenter()
        center.play(url: "/uploads/s/a.mp4", data: Data([0x00, 0x01, 0x02]))
        XCTAssertEqual(center.activeURL, "/uploads/s/a.mp4")
        XCTAssertNotNil(center.player)
        center.stopAll()
    }

    func testPlayingSameRowAgainIsNotARestart() {
        // Unlike audio, a second click on the playing row is a no-op: the
        // player on screen carries its own transport controls.
        let center = VideoCenter()
        center.play(url: "/uploads/s/a.mp4", data: Data([0x00]))
        let before = center.player.map(ObjectIdentifier.init)
        center.play(url: "/uploads/s/a.mp4", data: Data([0x00]))
        XCTAssertEqual(center.activeURL, "/uploads/s/a.mp4")
        XCTAssertEqual(center.player.map(ObjectIdentifier.init), before)
        center.stopAll()
    }

    func testSecondRowReplacesTheFirst() {
        let center = VideoCenter()
        center.play(url: "/uploads/s/one.mp4", data: Data([0x00]))
        let first = center.player.map(ObjectIdentifier.init)
        center.play(url: "/uploads/s/two.mp4", data: Data([0x00]))
        XCTAssertEqual(center.activeURL, "/uploads/s/two.mp4")
        XCTAssertNotEqual(center.player.map(ObjectIdentifier.init), first)
        XCTAssertNil(center.state(for: "/uploads/s/one.mp4").isReady ? true : nil)
        center.stopAll()
    }

    func testStopAllClearsEverything() {
        let center = VideoCenter()
        center.play(url: "/uploads/s/a.mp4", data: Data([0x00]))
        center.stopAll()
        XCTAssertNil(center.activeURL)
        XCTAssertNil(center.player)
        XCTAssertEqual(center.state(for: "/uploads/s/a.mp4"), VideoCenter.RowState())
    }

    func testGarbageIsReportedNotPlayed() async throws {
        // Bytes no demuxer accepts: the item fails asynchronously, the row
        // falls back to its download chip and the player is torn down.
        let center = VideoCenter()
        center.play(url: "/uploads/s/bad.mp4", data: Data([0x00, 0x01, 0x02]))
        var waited = 0.0
        while waited < 10 {
            if center.state(for: "/uploads/s/bad.mp4").loadFailed { break }
            try await Task.sleep(for: .milliseconds(100))
            waited += 0.1
        }
        XCTAssertTrue(center.state(for: "/uploads/s/bad.mp4").loadFailed)
        XCTAssertNil(center.activeURL)
        XCTAssertNil(center.player)
    }
}
