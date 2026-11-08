import XCTest
import AVFoundation
@testable import OnIt

@MainActor
final class AudioCenterTests: XCTestCase {
    /// A valid 0.2 s 440 Hz WAV, built in memory: the player must be fed
    /// something CoreAudio really decodes for the state machine to be real.
    private func makeWav(seconds: Double = 0.2) -> Data {
        let sampleRate = 8_000
        let count = Int(Double(sampleRate) * seconds)
        var samples = [Int16]()
        for i in 0..<count {
            samples.append(Int16(sin(2 * .pi * 440 * Double(i) / Double(sampleRate)) * 12_000))
        }
        var data = Data()
        func ascii(_ s: String) { data.append(s.data(using: .ascii)!) }
        func le32(_ v: UInt32) { var x = v.littleEndian; data.append(Data(bytes: &x, count: 4)) }
        func le16(_ v: UInt16) { var x = v.littleEndian; data.append(Data(bytes: &x, count: 2)) }
        ascii("RIFF"); le32(UInt32(36 + count * 2)); ascii("WAVE")
        ascii("fmt "); le32(16); le16(1); le16(1)
        le32(UInt32(sampleRate)); le32(UInt32(sampleRate * 2)); le16(2); le16(16)
        ascii("data"); le32(UInt32(count * 2))
        samples.withUnsafeBufferPointer { data.append(Data(buffer: $0)) }
        return data
    }

    override func tearDown() {
        AudioCenter().stopAll()
        super.tearDown()
    }

    func testTogglePlaysAndStops() {
        let center = AudioCenter()
        center.toggle(url: "/uploads/s/a.wav", data: makeWav())
        XCTAssertTrue(center.state(for: "/uploads/s/a.wav").isPlaying)
        XCTAssertEqual(center.activeURL, "/uploads/s/a.wav")

        center.toggle(url: "/uploads/s/a.wav", data: makeWav())  // same row again = stop
        XCTAssertFalse(center.state(for: "/uploads/s/a.wav").isPlaying)
        XCTAssertNil(center.activeURL)
    }

    func testSecondRowSilencesTheFirst() {
        let center = AudioCenter()
        center.toggle(url: "/uploads/s/one.wav", data: makeWav())
        center.toggle(url: "/uploads/s/two.wav", data: makeWav())
        XCTAssertFalse(center.state(for: "/uploads/s/one.wav").isPlaying)
        XCTAssertTrue(center.state(for: "/uploads/s/two.wav").isPlaying)
        center.stopAll()
    }

    func testGarbageIsReportedNotPlayed() {
        let center = AudioCenter()
        center.toggle(url: "/uploads/s/bad.wav", data: Data([0x00, 0x01, 0x02]))
        let state = center.state(for: "/uploads/s/bad.wav")
        XCTAssertFalse(state.isPlaying)
        XCTAssertTrue(state.loadFailed)
        XCTAssertNil(center.activeURL)
    }
}
