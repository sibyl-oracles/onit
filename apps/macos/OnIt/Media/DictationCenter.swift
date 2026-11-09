import AVFoundation
import Foundation
import Observation
import Speech

/// A spoken command recognized in a dictation stream. Every command is
/// anchored on the wake word "onit" — bare "execute", "run it", or "stop"
/// are ordinary words that appear in half of all task text, and treating
/// them as commands ate the dictated task (say "…and execute" and the whole
/// task was consumed as a command instead of sent).
///
/// The grammar is strict: anchor, then the command phrase, then nothing.
/// "onit run" fires; "onit run the tests" is a task addressed to onit and
/// passes through untouched.
enum VoiceCommand: Equatable {
    /// "onit, run" / "onit, run it" / "onit, execute" — send the composer.
    case run
    /// "onit, stop" / "onit, stop it" — barge in and stop the running task.
    case stop

    /// Normalize a dictation stream and peel a command off its tail.
    /// Returns the text with the command words removed, plus the command.
    ///
    /// Normalization folds the two things the recognizer is inconsistent
    /// about: case, punctuation, and the "onit"/"on it" homophone pair.
    /// Filler words are dropped so "onit, um, execute" still fires.
    static func consume(from text: String) -> (remaining: String, command: VoiceCommand)? {
        let words = normalize(text)
        guard !words.isEmpty, let anchor = anchorRange(in: words) else { return nil }

        // Everything before the anchor is the task; everything after must be
        // exactly the command phrase — extra words make it a task, not a
        // command ("onit execute this script" should run the script, not
        // send the draft). The anchor itself is command words too: it must
        // not ride into the sent message.
        let head = words[..<anchor.lowerBound]
        let tail = Array(words[anchor.upperBound...])

        switch tail {
        case ["stop"], ["stop", "it"]:
            return (head.joined(separator: " "), .stop)
        case ["run"], ["run", "it"], ["execute"]:
            return (head.joined(separator: " "), .run)
        default:
            return nil
        }
    }

    /// Range of the wake word. "onit" is spoken as one word but the
    /// recognizer often emits the English pair "on it", so both count. The
    /// last occurrence wins: the command follows the most recent anchor.
    private static func anchorRange(in words: [String]) -> Range<Int>? {
        for i in words.indices.reversed() {
            if words[i] == "onit" { return i..<i + 1 }
            if words[i] == "on", i + 1 < words.count, words[i + 1] == "it" { return i..<i + 2 }
        }
        return nil
    }

    /// Lowercase, split on anything that is not a letter or digit, and drop
    /// the filler words speech pads sentences with.
    private static func normalize(_ text: String) -> [String] {
        let fillers: Set<String> = ["um", "uh", "hey", "please"]
        return text.lowercased()
            .split(whereSeparator: { !$0.isLetter && !$0.isNumber })
            .map(String.init)
            .filter { !fillers.contains($0) }
    }
}

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
    ///
    /// Also ends the recognizer's buffer: the task that produced the words
    /// keeps its internal buffer, and without this it re-emits everything
    /// already heard as the next partial — the just-sent text would land
    /// back in the draft, looking like the send never happened. Ending the
    /// request flushes that buffer; the next press starts a fresh one.
    func reset() {
        transcript = ""
        partial = ""
        error = nil
        request?.endAudio()
        request = nil
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
    ///
    /// A final result is not always the whole story. The server path repeats
    /// everything recognized so far (cumulative), while on-device recognition
    /// finalizes per utterance — each final holds only the words since the
    /// last pause. Replacing unconditionally dropped everything dictated
    /// before the pause: say "execute" after a dictated task and the
    /// transcript shrank to just "execute", wiping the task from the draft.
    /// So: a final that starts with the transcript (the cumulative shape)
    /// replaces it; anything else continues it.
    func commitFinal(_ text: String) {
        partial = ""
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        let previous = Self.words(transcript)
        let incoming = Self.words(trimmed)
        let cumulative = incoming.count >= previous.count
            && incoming.prefix(previous.count).elementsEqual(previous)
        if cumulative {
            transcript = trimmed
        } else {
            transcript = transcript.isEmpty ? trimmed : transcript + " " + trimmed
        }
    }

    /// Lowercased words with punctuation stripped — the shape finals must be
    /// compared in, because the recognizer's casing and punctuation are not
    /// stable from one utterance to the next.
    private static func words(_ text: String) -> [String] {
        text.lowercased()
            .split(whereSeparator: { !$0.isLetter && !$0.isNumber })
            .map(String.init)
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
