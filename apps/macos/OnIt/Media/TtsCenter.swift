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
///
/// What is spoken is the server's *summary* of the answer, never the full
/// text: the summary is prefetched while the answer lands, and when it
/// cannot be produced the center stays silent rather than read the whole
/// reply aloud. Reading every word is the thing read-aloud exists to avoid;
/// a listener who wants the long version asks for it with the speak tool.
@MainActor
@Observable
final class TtsCenter: NSObject, AVAudioPlayerDelegate {
    /// Read from the toggle row; written only through `setEnabled`.
    private(set) var isEnabled: Bool = {
        UserDefaults.standard.bool(forKey: "onit.ttsEnabled")
    }()

    /// What read-aloud is doing right now. The composer shows it: the
    /// summary being prepared is a wait the listener should see, not
    /// silence they have to guess at.
    enum Phase: Equatable {
        /// Nothing is happening.
        case idle
        /// The spoken summary is being prepared — polling the prefetch.
        case summarizing
        /// A clip is actually sounding.
        case speaking
    }

    private(set) var phase: Phase = .idle

    /// True while a clip is actually sounding, so the toggle can show it.
    var isSpeaking: Bool { phase == .speaking }

    /// Called on every speaking-state transition — `true` when a clip
    /// starts sounding, `false` when it stops. Deliberately *not* fired
    /// while summarizing: the mic-mute gate cares about audio, not about
    /// preparation, and the recognizer must keep working while the
    /// summary is still being made.
    var onSpeakingChange: ((Bool) -> Void)?

    /// The player in flight, when one is.
    private var player: AVAudioPlayer?

    /// The in-flight fetch, cancelled by `stop()` and by a newer reply so
    /// a stale poll cannot resurrect "Summarizing…" after the user moved on.
    private var speakTask: Task<Void, Never>?

    /// How long the prefetch poll waits before giving up. The server starts
    /// the summary when the answer lands, but the model call behind it can
    /// take longer than a beat — long enough that the old three-second
    /// deadline expired and the full text got spoken instead.
    static let summaryPollTimeout: TimeInterval = 30

    func setEnabled(_ on: Bool) {
        isEnabled = on
        UserDefaults.standard.set(on, forKey: "onit.ttsEnabled")
        if !on { stop() }
    }

    /// The one place the phase flips with the callback fired — every
    /// transition into or out of *speaking* goes through here.
    private func setPhase(_ newPhase: Phase) {
        guard phase != newPhase else { return }
        let wasSpeaking = phase == .speaking
        phase = newPhase
        let isSpeakingNow = newPhase == .speaking
        if wasSpeaking != isSpeakingNow {
            onSpeakingChange?(isSpeakingNow)
        }
    }

    func stop() {
        speakTask?.cancel()
        speakTask = nil
        player?.stop()
        player = nil
        setPhase(.idle)
    }

    /// Speak one finished reply. Markdown is stripped here, mirroring the
    /// web UI's client-side strip: code blocks and link syntax read badly.
    ///
    /// When the server reported a prefetch job for this answer, the clip is
    /// already being made; the poll picks it up the moment it is done. When
    /// no prefetch exists — the server has summarize off — this reply is
    /// spoken verbatim on demand, which is the explicit opt-out.
    func speak(_ text: String, client: OnItClient, session: String? = nil, ttsId: Int? = nil) {
        guard isEnabled else { return }
        stop()
        let stripped = stripForSpeech(text)
        guard !stripped.isEmpty else { return }
        // Visible immediately, synchronously: the wait between the answer
        // landing and the first word is exactly what the label explains.
        if ttsId != nil {
            phase = .summarizing
        }
        speakTask = Task { await self.speakStripped(stripped, client: client, session: session, ttsId: ttsId) }
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
        setPhase(.speaking)
    }

    private func speakStripped(
        _ text: String, client: OnItClient, session: String? = nil, ttsId: Int? = nil
    ) async {
        do {
            var data: Data?
            if let session, let ttsId {
                // The prefetch has a head start on this loop, but the model
                // call behind the summary can take longer than a beat: wait
                // for it rather than cutting to the full text.
                let deadline = Date().addingTimeInterval(Self.summaryPollTimeout)
                while Date() < deadline {
                    try Task.checkCancellation()
                    switch await client.ttsSummary(ttsId, session: session) {
                    case let .ready(wav):
                        data = wav
                    case .pending:
                        try await Task.sleep(nanoseconds: 250_000_000)
                        continue
                    case .missing:
                        break
                    }
                    break
                }
            }
            let wav: Data
            if let data {
                wav = data
            } else if ttsId != nil {
                // No summary arrived — the job failed or the server could
                // not condense. Stay silent: reading the full answer aloud
                // is the exact thing read-aloud exists to avoid. The user
                // can read the reply on screen, or ask the speak tool for
                // the long version.
                setPhase(.idle)
                return
            } else {
                // No prefetch was ever promised (server summarize off): the
                // verbatim clip is what the opt-out asks for.
                wav = try await client.tts(text, session: session, summarize: false)
            }
            try Task.checkCancellation()
            let player = try AVAudioPlayer(data: wav)
            player.delegate = self
            guard player.play() else {
                setPhase(.idle)
                return
            }
            self.player = player
            setPhase(.speaking)
        } catch {
            // A reply the user can already read does not need an error
            // dialog about why it could not also be heard. Cancellation
            // lands here too: the phase was already reset by stop().
        }
    }

    // MARK: AVAudioPlayerDelegate

    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor in
            self.player = nil
            self.setPhase(.idle)
        }
    }
}
