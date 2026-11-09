import AVFoundation
import Foundation
import Observation
import Speech

/// Dictation for the composer, backed by Apple's Speech framework — the same
/// engine macOS Dictation uses, so spoken text lands in the draft field the
/// way the system's own dictation would.
///
/// The mic button is an *alternative* input, not a mode: the draft stays a
/// plain editable field, and dictation appends to whatever is already there.
/// The append base is snapshotted when listening starts, so a half-typed
/// sentence and a dictated sentence compose instead of overwrite.
///
/// Permission is the part that must not surprise: microphone first, then
/// speech recognition, each with its own System Settings prompt. A denial is
/// a state, not an error — the composer shows where to flip the switch,
/// because retrying cannot fix it.
@MainActor
@Observable
final class DictationCenter: NSObject, SFSpeechRecognizerDelegate {
    /// Everything the composer row needs, in one Equatable value so the view
    /// can observe a single change and order its reactions deterministically
    /// (capture the append base *before* applying new text).
    struct Snapshot: Equatable {
        var isListening = false
        var currentText = ""
        var error: String?
        var isDenied = false
    }

    var snapshot: Snapshot {
        Snapshot(
            isListening: isListening,
            currentText: currentText,
            error: error,
            isDenied: isDenied
        )
    }

    /// Finalized utterances plus the in-flight hypothesis, joined the way a
    /// person would have typed them. Speech finalizes on a pause, so one
    /// breath between sentences must not look like two dictation sessions.
    var currentText: String {
        switch (transcript.isEmpty, partial.isEmpty) {
        case (true, true): return ""
        case (true, false): return partial
        case (false, true): return transcript
        default: return transcript + " " + partial
        }
    }

    private(set) var isListening = false
    private(set) var error: String?
    /// True when a permission was refused: the fix lives in System Settings.
    private(set) var isDenied = false

    private var transcript = ""
    private var partial = ""
    private var recognizer: SFSpeechRecognizer?
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private let engine = AVAudioEngine()

    func toggle() {
        Task { await toggleAsync() }
    }

    private func toggleAsync() async {
        if isListening {
            stop()
        } else {
            await start()
        }
    }

    /// Begin listening. Any previous error is cleared: a fresh press is a
    /// fresh attempt, and a stale failure message next to a live mic reads
    /// like the failure is still happening.
    private func start() async {
        error = nil
        isDenied = false

        guard await ensurePermissions() else { return }

        // Match the user's language, falling back to US English; a locale
        // Speech cannot handle yields nil rather than a broken recognizer.
        let recognizer = SFSpeechRecognizer(locale: .current)
            ?? SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
        guard let recognizer, recognizer.isAvailable else {
            error = "Speech recognition is unavailable for this language."
            return
        }
        recognizer.delegate = self
        self.recognizer = recognizer

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        request.addsPunctuation = true
        // Prefer on-device transcription: it is private and works offline.
        // Not forced everywhere, because some locales/languages still only
        // have the server path — a hard requirement there would break
        // dictation entirely rather than degrade it.
        if recognizer.supportsOnDeviceRecognition {
            request.requiresOnDeviceRecognition = true
        }
        self.request = request

        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        do {
            input.installTap(onBus: 0, bufferSize: 2048, format: format) { buffer, _ in
                request.append(buffer)
            }
            engine.prepare()
            try engine.start()
        } catch {
            input.removeTap(onBus: 0)
            self.request = nil
            self.error = "Microphone unavailable: \(error.localizedDescription)"
            return
        }

        isListening = true
        task = recognizer.recognitionTask(with: request) { [weak self] result, taskError in
            Task { @MainActor in self?.handle(result: result, error: taskError) }
        }
    }

    /// Stop listening but keep the text: the words already belong to the
    /// draft, and a second press continues from there.
    func stop() {
        guard isListening else { return }
        isListening = false
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        request?.endAudio()
        // finish() flushes one last (usually final) result for the tail the
        // user spoke just before releasing the button.
        task?.finish()
        task = nil
        request = nil
    }

    /// Drop accumulated text and any error. Called when the draft leaves
    /// (send) so a stale transcript cannot resurface in the next dictation.
    func reset() {
        transcript = ""
        partial = ""
        error = nil
    }

    // MARK: - Recognition callbacks

    private func handle(result: SFSpeechRecognitionResult?, error: Error?) {
        if let result {
            let text = result.bestTranscription.formattedString
            if result.isFinal {
                commitFinal(text)
            } else {
                updatePartial(text)
            }
        }
        if let error {
            let ns = error as NSError
            // 1101 = no speech detected (a silence is normal life, not a
            // failure — macOS dictation ends the session the same way);
            // 216 = cancelled, which is our own stop() arriving late.
            // kAFAssistantErrorDomain is not surfaced to Swift by this SDK;
            // the domain string is the stable identifier.
            if ns.domain == "kAFAssistantErrorDomain", ns.code == 1101 || ns.code == 216 {
                if isListening { stop() }
                return
            }
            fail("Dictation failed: \(error.localizedDescription)")
        }
    }

    /// State transitions the recognition callback drives; internal so tests
    /// can exercise the accumulator without a microphone.
    func commitFinal(_ text: String) {
        transcript = text
        partial = ""
    }

    func updatePartial(_ text: String) {
        partial = text
    }

    private func fail(_ message: String) {
        error = message
        if isListening { stop() }
    }

    // MARK: - Permissions

    private func ensurePermissions() async -> Bool {
        // Microphone first: without it the engine start fails opaquely, and
        // the user deserves the specific missing switch, not a generic error.
        guard await requestMic() else {
            isDenied = true
            error = "Microphone access is off for OnIt — allow it in System Settings ▸ Privacy & Security ▸ Microphone."
            return false
        }
        guard await requestSpeech() else {
            isDenied = true
            error = "Speech recognition is off for OnIt — allow it in System Settings ▸ Privacy & Security ▸ Speech Recognition."
            return false
        }
        return true
    }

    private func requestMic() async -> Bool {
        await withCheckedContinuation { continuation in
            AVAudioApplication.requestRecordPermission { granted in
                continuation.resume(returning: granted)
            }
        }
    }

    private func requestSpeech() async -> Bool {
        switch SFSpeechRecognizer.authorizationStatus() {
        case .authorized:
            return true
        case .denied, .restricted:
            return false
        case .notDetermined:
            return await withCheckedContinuation { continuation in
                SFSpeechRecognizer.requestAuthorization { status in
                    continuation.resume(returning: status == .authorized)
                }
            }
        @unknown default:
            return false
        }
    }

    // MARK: SFSpeechRecognizerDelegate

    nonisolated func speechRecognizer(
        _ speechRecognizer: SFSpeechRecognizer,
        availabilityDidChange available: Bool
    ) {
        guard !available else { return }
        Task { @MainActor in self.fail("Speech recognition became unavailable.") }
    }
}
