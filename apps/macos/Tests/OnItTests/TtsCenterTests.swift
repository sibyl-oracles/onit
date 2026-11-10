import XCTest
import AVFoundation
@testable import OnIt

@MainActor
final class TtsCenterTests: XCTestCase {
    /// A valid 0.2 s 440 Hz WAV, built in memory — the same fixture
    /// AudioCenterTests uses, because the player must be fed something
    /// CoreAudio really decodes for the state machine to be real.
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

    override func setUp() {
        UserDefaults.standard.removeObject(forKey: "onit.ttsEnabled")
        super.setUp()
    }

    override func tearDown() {
        TtsCenter().stop()
        UserDefaults.standard.removeObject(forKey: "onit.ttsEnabled")
        super.tearDown()
    }

    func testDisabledByDefault() {
        XCTAssertFalse(TtsCenter().isEnabled)
    }

    func testEnabledPersists() {
        let center = TtsCenter()
        center.setEnabled(true)
        XCTAssertTrue(center.isEnabled)
        XCTAssertEqual(UserDefaults.standard.bool(forKey: "onit.ttsEnabled"), true)
        center.setEnabled(false)
        XCTAssertFalse(center.isEnabled)
        XCTAssertEqual(UserDefaults.standard.bool(forKey: "onit.ttsEnabled"), false)
    }

    func testDisablingStopsPlayback() {
        let center = TtsCenter()
        center.setEnabled(true)
        center.playForTest(makeWav())
        XCTAssertTrue(center.isSpeaking)
        center.setEnabled(false)
        XCTAssertFalse(center.isSpeaking)
    }

    func testSecondReplySilencesTheFirst() {
        let center = TtsCenter()
        center.setEnabled(true)
        center.playForTest(makeWav())
        XCTAssertTrue(center.isSpeaking)
        center.playForTest(makeWav())
        XCTAssertTrue(center.isSpeaking)  // the new one, not two at once
        center.stop()
        XCTAssertFalse(center.isSpeaking)
    }

    func testMarkdownIsStrippedBeforeSpeaking() {
        let center = TtsCenter()
        let stripped = center.stripForSpeech(
            "# Title\n\nSee [the docs](https://x.y) and ```\ncode()\n```\n\n**bold** _it_."
        )
        XCTAssertFalse(stripped.contains("#"))
        XCTAssertFalse(stripped.contains("](http"))
        XCTAssertFalse(stripped.contains("code()"))
        XCTAssertFalse(stripped.contains("**"))
        XCTAssertTrue(stripped.contains("the docs"))
        XCTAssertTrue(stripped.contains("bold"))
    }

    func testEmptyAfterStripIsNotSpoken() {
        let center = TtsCenter()
        center.setEnabled(true)
        // A reply that is only a code fence has nothing to say out loud.
        center.speak("```python\nprint(1)\n```", client: OnItClient(baseURL: URL(string: "http://127.0.0.1:1")!))
        XCTAssertFalse(center.isSpeaking)
    }

    // MARK: - The speaking-state callback: the dictation center mutes its
    // recognizer on it, so every transition must fire it exactly once.

    func testSpeakingChangeFiresOnStart() {
        let center = TtsCenter()
        var fired: [Bool] = []
        center.onSpeakingChange = { fired.append($0) }
        center.setEnabled(true)
        center.playForTest(makeWav())
        XCTAssertEqual(fired, [true])
    }

    func testSpeakingChangeFiresOnStop() {
        let center = TtsCenter()
        var fired: [Bool] = []
        center.onSpeakingChange = { fired.append($0) }
        center.setEnabled(true)
        center.playForTest(makeWav())
        center.stop()
        XCTAssertEqual(fired, [true, false])
    }

    func testSpeakingChangeFiresOnDisableMidClip() {
        let center = TtsCenter()
        var fired: [Bool] = []
        center.onSpeakingChange = { fired.append($0) }
        center.setEnabled(true)
        center.playForTest(makeWav())
        center.setEnabled(false)  // stops the clip and unmutes dictation
        XCTAssertEqual(fired, [true, false])
    }

    func testSpeakingChangeDoesNotFireWithoutTransition() {
        let center = TtsCenter()
        var fired: [Bool] = []
        center.onSpeakingChange = { fired.append($0) }
        center.stop()  // not speaking; stop must not fire a spurious false
        XCTAssertEqual(fired, [])
        center.setEnabled(true)
        center.playForTest(makeWav())
        center.playForTest(makeWav())  // new clip while speaking: no false
        XCTAssertEqual(fired, [true])
        center.stop()
        XCTAssertEqual(fired, [true, false])
    }
}
