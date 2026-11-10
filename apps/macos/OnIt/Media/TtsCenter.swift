import AVFoundation
import Foundation
import Observation

/// Speaks the agent's replies when the user has turned the toggle on.
///
/// One player at a time, app-wide — the same rule AudioCenter follows for
/// attachments: two overlapping voices is never what a person wanted from a
/// chat transcript. A new reply stops the clip in flight, and stopping the
/// toggle stops everything.
///
/// The bytes come from the server's local Kokoro model (`/api/tts`), so a
/// reply spoken here costs no cloud call and no key. Synthesis is not
/// streamed: the whole clip arrives before playback starts, which is why
/// this is fed from a *finished* answer rather than the token stream.
@MainActor
@Observable
final class TtsCenter: NSObject, AVAudioPlayerDelegate {
    /// Read from the toggle row; written only through `setEnabled`.
    private(set) var isEnabled: Bool = {
        UserDefaults.standard.bool(forKey: "onit.ttsEnabled")
    }()

    /// True while a clip is actually sounding, so the toggle can show it.
    private(set) var isSpeaking = false

    /// Called on every speaking-state transition — `true` when a clip
    /// starts sounding, `false` when it stops. The dictation center
    /// observes this to mute its recognizer for the duration: a mic that
    /// hears the app's own voice transcribes it into the draft.
    var onSpeakingChange: ((Bool) -> Void)?

    private var player: AVAudioPlayer?

    func setEnabled(_ on: Bool) {
        isEnabled = on
        UserDefaults.standard.set(on, forKey: "onit.ttsEnabled")
        if !on { stop() }
    }

    /// The one place isSpeaking flips with the callback fired — every
    /// transition (start, stop, disable mid-clip) goes through here.
    private func setSpeaking(_ speaking: Bool) {
        guard isSpeaking != speaking else { return }
        isSpeaking = speaking
        onSpeakingChange?(speaking)
    }

    func stop() {
        player?.stop()
        player = nil
        setSpeaking(false)
    }

    /// Speak one finished reply. Markdown is stripped here, mirroring the
    /// web UI's client-side strip: code blocks and link syntax read badly.
    func speak(_ text: String, client: OnItClient) {
        guard isEnabled else { return }
        stop()
        let stripped = stripForSpeech(text)
        guard !stripped.isEmpty else { return }
        Task { await self.speakStripped(stripped, client: client) }
    }

    /// The text actually sent to the synthesizer. Internal so tests can
    /// check the strip without a server behind the call.
    func stripForSpeech(_ text: String) -> String {
        text
            .replacingOccurrences(
                of: "```[\\s\\S]*?```",
                with: " ",
                options: .regularExpression
            )
            .replacingOccurrences(
                of: "!?\\[([^\\]]*)\\]\\([^)]*\\)",
                with: "$1",
                options: .regularExpression
            )
            .replacingOccurrences(of: "[#*_>`~|]", with: " ", options: .regularExpression)
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Feed the player directly. Test-only in spirit — production callers go
    /// through `speak`, which fetches from the server — but kept non-private
    /// so the state machine is testable without a live TTS endpoint.
    func playForTest(_ data: Data) {
        guard let player = try? AVAudioPlayer(data: data) else { return }
        player.delegate = self
        guard player.play() else { return }
        self.player = player
        setSpeaking(true)
    }

    private func speakStripped(_ text: String, client: OnItClient) async {
        do {
            let data = try await client.tts(text)
            let player = try AVAudioPlayer(data: data)
            player.delegate = self
            guard player.play() else { return }
            self.player = player
            setSpeaking(true)
        } catch {
            // A reply the user can already read does not need an error
            // dialog about why it could not also be heard.
        }
    }

    // MARK: AVAudioPlayerDelegate

    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor in
            self.player = nil
            self.setSpeaking(false)
        }
    }
}
