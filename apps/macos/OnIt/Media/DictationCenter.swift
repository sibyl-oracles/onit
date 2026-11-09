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
/// The anchored grammar is strict: anchor, then the command phrase, then
/// nothing. "onit run" fires; "onit run the tests" is a task addressed to
/// onit and passes through untouched.
///
/// There is a second, pause-delimited form: once an utterance has finalized
/// (the user paused), a follow-up utterance that is *exactly* a command
/// phrase — "run", "go", "execute", "stop", "clear" — fires without the
/// anchor. The
/// pause is what separates command from task text: a sentence that merely
/// ends with one of these words ("went for a run") is one utterance, not an
/// utterance plus a command, and passes through untouched.
///
/// "Finalized" is produced by two paths, because on-device recognition does
/// not guarantee the first: the recognizer's own `isFinal` when it emits one,
/// and a silence watchdog that commits the hypothesis after it has sat
/// unchanged for `pauseInterval` — on-device sessions often deliver only
/// partials until endAudio, and without the watchdog the bare form would
/// wait forever for a final that never comes.
enum VoiceCommand: Equatable {
    /// "onit, run" / "onit, run it" / "onit, execute" / "onit, go" — send.
    case run
    /// "onit, stop" / "onit, stop it" — barge in and stop the running task.
    case stop
    /// "onit, clear" / "onit, discard" — wipe the draft and start over.
    /// The view layer decides what a clear wipes; the center only peels
    /// the command and keeps the task text, like `stop`.
    case clear

    /// The phrases each command accepts, in normalized form.
    private static let runPhrases: Set<[String]> = [["run"], ["run", "it"], ["execute"], ["go"]]
    private static let stopPhrases: Set<[String]> = [["stop"], ["stop", "it"]]
    private static let clearPhrases: Set<[String]> = [
        ["clear"], ["clear", "it"], ["discard"], ["discard", "it"], ["start", "over"],
    ]

    /// Normalize a dictation stream and peel a command off its tail.
    /// Returns the text with the command words removed, the command, and how
    /// many normalized words the command consumed — the raw-text strip in
    /// `strippingLastWords` needs the count.
    ///
    /// Normalization folds the two things the recognizer is inconsistent
    /// about: case, punctuation, and the "onit"/"on it" homophone pair.
    /// Filler words are dropped so "onit, um, execute" still fires.
    ///
    /// `utterance` is the text of the utterance the recognizer just
    /// finalized, when there is one; it unlocks the pause-delimited bare
    /// form. On a partial there is no utterance boundary yet, so bare words
    /// never fire mid-stream.
    static func consume(
        from text: String,
        utterance: String? = nil
    ) -> (remaining: String, command: VoiceCommand, consumed: Int)? {
        let words = normalize(text)
        guard !words.isEmpty else { return nil }

        if let anchor = anchorRange(in: words) {
            // Everything before the anchor is the task; everything after must
            // be exactly the command phrase — extra words make it a task, not
            // a command ("onit execute this script" should run the script,
            // not send the draft). The anchor itself is command words too: it
            // must not ride into the sent message.
            let head = words[..<anchor.lowerBound]
            let tail = Array(words[anchor.upperBound...])
            if let command = commandPhrase(tail) {
                return (head.joined(separator: " "), command, words.count - head.count)
            }
        }

        // No anchored command: try the pause-delimited bare form.
        return bare(stream: words, utterance: utterance)
    }

    /// The pause-delimited form: the finalized utterance is exactly a
    /// command phrase, and it sits at the tail of the stream — a newer
    /// partial in flight breaks the suffix and vetoes the fire.
    private static func bare(
        stream: [String],
        utterance: String?
    ) -> (remaining: String, command: VoiceCommand, consumed: Int)? {
        guard let utterance, !utterance.isEmpty else { return nil }
        let phrase = normalize(utterance)
        guard stream.count >= phrase.count,
              Array(stream.suffix(phrase.count)) == phrase,
              let command = commandPhrase(phrase)
        else { return nil }
        let head = stream.prefix(stream.count - phrase.count)
        return (head.joined(separator: " "), command, phrase.count)
    }

    private static func commandPhrase(_ words: [String]) -> VoiceCommand? {
        if stopPhrases.contains(words) { return .stop }
        if runPhrases.contains(words) { return .run }
        if clearPhrases.contains(words) { return .clear }
        return nil
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

    /// Remove the command's words from the raw text, preserving the raw
    /// casing and punctuation of everything before them — the stripped text
    /// is what gets sent or kept in the draft, and it must read the way the
    /// user spoke it ("Summarize chapter 3.", not "summarize chapter 3").
    ///
    /// The command is always a raw suffix (both grammars end with it), so
    /// the walk goes backwards over whitespace tokens, counting normalized
    /// words until the consumed count is met. Filler tokens count as zero
    /// words and are cut too — "onit, um, run" is three raw tokens, two
    /// words. Returns nil when the tokens do not align; the caller then
    /// keeps the text as-is rather than corrupt it.
    static func strippingLastWords(_ consumed: Int, from raw: String) -> String? {
        guard consumed > 0 else { return nil }
        let tokens = raw.split(separator: " ", omittingEmptySubsequences: true).map(String.init)
        var seen = 0
        var cut = 0
        for token in tokens.reversed() {
            let count = normalize(token).count
            if seen + count > consumed { return nil }
            seen += count
            cut += 1
            if seen == consumed { break }
        }
        guard seen == consumed else { return nil }
        var kept = tokens.dropLast(cut).joined(separator: " ")
        // The punctuation that introduced the command ("wait, onit stop")
        // dangles once the command is gone; separator marks go with it.
        // Terminal punctuation ("Summarize chapter 3.") is kept.
        while let last = kept.last, ",;:-".contains(last) {
            kept = String(kept.dropLast()).trimmingCharacters(in: .whitespaces)
        }
        return kept
    }
}

/// A command the matcher found in the stream, waiting for the composer to
/// act on it. Carries the word count the command consumed so the raw strip
/// can remove exactly those words.
struct PendingCommand: Equatable {
    let command: VoiceCommand
    let consumedWords: Int
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
        /// A spoken command waiting for the composer to act. Consumed
        /// exactly once via `takeCommand()`.
        var pendingCommand: PendingCommand?
    }

    var snapshot: Snapshot {
        Snapshot(
            isListening: isListening,
            currentText: currentText,
            error: error,
            isDenied: isDenied,
            pendingCommand: pendingCommand
        )
    }

    /// Finalized utterances plus the in-flight hypothesis, joined the way a
    /// person would have typed them. Speech finalizes on a pause, so one
    /// breath between sentences must not look like two dictation sessions.
    ///
    /// The join also refuses to paint a word twice. The recognizer's stream
    /// repeats what it has already committed — a cumulative partial grows
    /// past the transcript, and after its head is revised (the server path
    /// re-scores earlier words) it re-speaks the transcript's tail — so a
    /// naive transcript + partial read showed every committed word twice,
    /// growing with each pause. The preview shows only what is new.
    var currentText: String {
        switch (transcript.isEmpty, partial.isEmpty) {
        case (true, true): return ""
        case (true, false): return partial
        case (false, true): return transcript
        default: break
        }
        let p = Self.words(transcript)
        let l = Self.words(partial)
        // Cumulative partial: the stream repeats the transcript and grows
        // past it. Preview the delta only.
        if l.count > p.count,
           l.prefix(p.count).elementsEqual(p),
           let delta = Self.rawTrailingTokens(l.count - p.count, in: partial) {
            return transcript + " " + delta
        }
        // Cumulative with a revised head: the stream no longer begins where
        // the transcript begins, but it re-speaks the transcript's tail and
        // adds a short extension past it. Preview the extension only.
        if let c = Self.driftExtension(previous: p, incoming: l) {
            if c == 0 { return transcript }
            if let ext = Self.rawTrailingTokens(c, in: partial) {
                return transcript + " " + ext
            }
        }
        return transcript + " " + partial
    }

    private(set) var isListening = false
    private(set) var error: String?
    /// True when a permission was refused: the fix lives in System Settings.
    private(set) var isDenied = false

    private var transcript = ""
    private var partial = ""
    private var pendingCommand: PendingCommand?
    private var recognizer: SFSpeechRecognizer?
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    /// Bumped on every restart and stop. Recognition callbacks carry the
    /// generation they belong to, and stale ones are dropped: a flushed
    /// request still delivers its final, and that final must not land in a
    /// fresh transcript — it resurrected the sent words into the draft,
    /// looking like the send never happened.
    private var generation = 0
    /// How long the hypothesis must sit unchanged before it counts as a
    /// pause. The user's contract is one second; a little under a breath.
    static let pauseInterval: TimeInterval = 1.0
    /// The pause watchdog. Rescheduled on every partial change; when it
    /// fires, the stable hypothesis is committed as if the recognizer had
    /// finalized it, which is what unlocks the bare command form on the
    /// on-device path (that path often emits no mid-stream finals).
    private var pauseWork: DispatchWorkItem?
    /// A fresh engine per listening session. AVAudioEngine's graph is not
    /// reliably reusable across start/stop cycles on macOS 26 — re-preparing
    /// a stopped engine raised an NSException that Swift's do/catch cannot
    /// intercept, aborting the process. Rebuilding is cheap and always sound.
    private var engine = AVAudioEngine()

    func toggle() {
        // Hopping through the main queue before spawning the task re-enters
        // the runloop outside SwiftUI's event dispatch. During debugging the
        // direct `Task {}` hung in the permission round-trip when invoked
        // from an accessibility-driven press; the hop makes both paths —
        // mouse and accessibility — behave identically.
        DispatchQueue.main.async {
            Task { @MainActor in await self.toggleAsync() }
        }
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
        pendingCommand = nil

        let ok = await ensurePermissions()
        guard ok else { return }

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

        let request = makeRequest()
        self.request = request

        engine = AVAudioEngine()
        do {
            // Materialize the input node BEFORE prepare: on macOS 26,
            // prepare() with no nodes attached raises an NSException
            // ('inputNode != nullptr || outputNode != nullptr') that Swift
            // cannot catch — touching the input node first gives the graph
            // its node.
            _ = engine.inputNode
            engine.prepare()
            try engine.start()
        } catch {
            self.request = nil
            self.error = "Microphone unavailable: \(error.localizedDescription)"
            return
        }

        isListening = true
        beginTask(with: request)
    }

    /// A fresh recognition request with the settings every request in this
    /// class shares — partials on, punctuation on, on-device when the
    /// recognizer supports it (private, offline; not forced everywhere
    /// because some locales still only have the server path — a hard
    /// requirement there would break dictation rather than degrade it).
    private func makeRequest() -> SFSpeechAudioBufferRecognitionRequest {
        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        request.addsPunctuation = true
        if recognizer?.supportsOnDeviceRecognition == true {
            request.requiresOnDeviceRecognition = true
        }
        return request
    }

    /// Attach `request` to the microphone and start recognizing. The tap is
    /// (re)installed so it feeds this request — a restart swaps requests
    /// under a running engine, and a tap that still held the old one would
    /// keep feeding audio to the request that was just flushed.
    private func beginTask(with request: SFSpeechAudioBufferRecognitionRequest) {
        let input = engine.inputNode
        input.removeTap(onBus: 0)
        input.installTap(onBus: 0, bufferSize: 2048, format: input.outputFormat(forBus: 0)) { buffer, _ in
            request.append(buffer)
        }
        generation += 1
        let gen = generation
        task = recognizer?.recognitionTask(with: request) { [weak self] result, taskError in
            Task { @MainActor in self?.handle(result: result, error: taskError, generation: gen) }
        }
    }

    /// Restart recognition in place: fresh request, fresh task, same running
    /// engine. Used after a send or a spoken command — the request that
    /// produced the words keeps them in its internal buffer, and without a
    /// restart it re-emits everything already heard as the next partial, so
    /// the just-sent text lands back in the draft. The old task's last
    /// callback is dropped by the generation guard.
    private func restart() {
        guard isListening, recognizer != nil else { return }
        cancelPauseWatchdog()
        request?.endAudio()
        task?.finish()
        task = nil
        let request = makeRequest()
        self.request = request
        beginTask(with: request)
    }

    /// Flush the recognizer's buffer without touching the accumulated text —
    /// the companion to `takeCommand()` for commands that must not clear the
    /// draft (a spoken stop keeps the task text for the next "onit, run").
    func flush() {
        guard isListening, recognizer != nil else { return }
        restart()
    }

    /// Stop listening but keep the text: the words already belong to the
    /// draft, and a second press continues from there.
    func stop() {
        guard isListening else { return }
        isListening = false
        pendingCommand = nil
        cancelPauseWatchdog()
        // Bump the generation before tearing down: the cancelled task's last
        // callbacks (an error 216, or a final for the tail just spoken) must
        // not land — the tail is already in the partial and the draft, and a
        // late final would append it a second time.
        generation += 1
        // Retire the recognition task explicitly before dropping it.
        // Releasing an SFSpeechRecognitionTask that is still running — no
        // finish(), no cancel() — crashed the NEXT main-actor dispatch with
        // heap corruption (objc_opt_class / swift_getObjectType under
        // ButtonAction): the abandoned session's callbacks landed in memory
        // that had already moved on. cancel() barges in without a final,
        // which is what a stop wants.
        request?.endAudio()
        task?.cancel()
        task = nil
        request = nil
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
    }

    /// Drop accumulated text and any error. Called when the draft leaves
    /// (send) so a stale transcript cannot resurface in the next dictation.
    ///
    /// While listening this restarts recognition rather than only clearing
    /// the accumulator: the request that produced the words keeps its
    /// internal buffer, and without a restart it re-emits everything already
    /// heard as the next partial — the just-sent text would land back in
    /// the draft, looking like the send never happened.
    func reset() {
        transcript = ""
        partial = ""
        pendingCommand = nil
        error = nil
        guard isListening, recognizer != nil else {
            generation += 1
            request?.endAudio()
            task = nil
            request = nil
            return
        }
        restart()
    }

    // MARK: - Recognition callbacks

    private func handle(result: SFSpeechRecognitionResult?, error: Error?, generation gen: Int) {
        // A callback from a flushed generation carries words the accumulator
        // no longer wants — the request was restarted out from under it.
        guard gen == self.generation else { return }
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
    ///
    /// Continuation has a second cumulative shape, the one behind the
    /// reported repetition bug: once the stream's head is revised — the
    /// server re-scores and drops or rewords its earliest words — the final
    /// no longer *starts* with the transcript, so the plain append fired and
    /// the whole re-spoken tail landed a second time. Every pause then grew
    /// the draft by everything dictated since the drift: "Hello Find the
    /// cheapest accommodation in Tokyo Find the cheapest accommodation in
    /// Tokyo for December …". When a final instead re-speaks the
    /// transcript's tail (at least 5 words, aligned at the transcript's end)
    /// and adds only a short extension past it, only the extension is
    /// appended; a pure re-emission of the tail lands nothing.
    ///
    /// A finalized utterance is also a command boundary — this is where the
    /// pause-delimited bare form becomes legal ("run" as its own utterance
    /// after the task finalized). For a cumulative final the utterance is
    /// the delta past the previous transcript: the words the recognizer
    /// added this time.
    func commitFinal(_ text: String) {
        partial = ""
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        let previous = Self.words(transcript)
        let incoming = Self.words(trimmed)
        guard !incoming.isEmpty else {
            setPendingCommand(utterance: nil)
            return
        }
        let previousCount = previous.count

        // Cumulative: the stream repeated everything and grew.
        if incoming.count >= previousCount,
           incoming.prefix(previousCount).elementsEqual(previous) {
            transcript = trimmed
            setPendingCommand(
                utterance: incoming.dropFirst(previousCount).joined(separator: " ")
            )
            return
        }

        // A pure echo of any length: the final repeats the transcript's own
        // tail and adds nothing (a flushed request replaying committed
        // words, or the real final for audio the watchdog already
        // committed). Land nothing. Command-shaped text is exempt: a bare
        // "run" after a task ending in "run" is a command, not an echo.
        if incoming.count < previousCount,
           previous.suffix(incoming.count).elementsEqual(incoming),
           !Self.isCommandShaped(trimmed) {
            setPendingCommand(utterance: nil)
            return
        }

        // Cumulative with a revised head: the final re-speaks the
        // transcript's tail and extends it. Append the extension only.
        if let c = Self.driftExtension(previous: previous, incoming: incoming) {
            if c == 0 {
                // A pure re-emission of the transcript's own tail — a
                // flushed request replaying words already committed.
                // Nothing new to land; re-check commands so no stale
                // pending survives.
                setPendingCommand(utterance: nil)
                return
            }
            if let ext = Self.rawTrailingTokens(c, in: trimmed) {
                transcript = transcript.isEmpty ? ext : transcript + " " + ext
                setPendingCommand(utterance: ext)
                return
            }
            // Raw tokens did not align with the word count — a hyphenated
            // or punctuated tangle the slice cannot express. Fall through
            // to the plain append rather than corrupt the text.
        }

        // A genuine new utterance.
        transcript = transcript.isEmpty ? trimmed : transcript + " " + trimmed
        setPendingCommand(utterance: trimmed)
    }

    /// Record the command the just-committed text may carry, exactly once.
    private func setPendingCommand(utterance: String?) {
        pendingCommand = VoiceCommand.consume(from: transcript, utterance: utterance).map {
            PendingCommand(command: $0.command, consumedWords: $0.consumed)
        }
    }

    /// The raw token(s) carrying the last `n` normalized words of `raw` —
    /// the mirror of `strippingLastWords` for appending instead of cutting.
    /// Used to extend the transcript with only the new tail of a cumulative
    /// final, keeping the raw casing and punctuation of the fresh words.
    /// Returns nil when the raw tokens do not align with the word count
    /// (a fused token like "on,it" cannot be split without corrupting it).
    static func rawTrailingTokens(_ n: Int, in raw: String) -> String? {
        guard n > 0 else { return "" }
        let tokens = raw.split(separator: " ", omittingEmptySubsequences: true).map(String.init)
        var seen = 0
        var cut = 0
        for token in tokens.reversed() {
            let count = Self.words(token).count
            if count == 0 { continue }
            if seen + count > n { return nil }
            seen += count
            cut += 1
            if seen == n { break }
        }
        guard seen == n else { return nil }
        return tokens.suffix(cut).joined(separator: " ")
    }

    /// How many words of `incoming` extend `previous` when `incoming` is a
    /// cumulative re-emission whose head has drifted — the server re-scores
    /// its earliest words, so the final no longer starts where the
    /// transcript starts, but it still re-speaks the transcript's tail and
    /// grows past it. Returns the extension count (0 for a pure re-emission
    /// of the tail), or nil when the shapes do not match this pattern and
    /// the caller should treat the final as a fresh utterance.
    ///
    /// The guard is deliberately strict, because every false positive eats
    /// real words: at least 5 words must re-align (a short tail matches too
    /// much by chance), the alignment must sit at the transcript's end, and
    /// the extension past it is capped at half the final's length — a
    /// genuinely new sentence is longer than the echo that precedes it.
    static func driftExtension(previous: [String], incoming: [String]) -> Int? {
        let n = incoming.count
        // The echo must re-align (5 words) and leave something to extend.
        guard n > 5, !previous.isEmpty else { return nil }
        // Longest overlap first: the echo is the maximal prefix of incoming
        // that matches the transcript's suffix; the extension is the rest.
        let maxK = min(previous.count, n)
        for k in stride(from: maxK, through: 5, by: -1) {
            let ext = n - k
            // The echo dominates: a "new" utterance longer than the echo is
            // a fresh sentence that happens to open with the same words.
            if ext * 2 >= n { break }
            if Array(previous.suffix(k)) == Array(incoming.prefix(k)) {
                return ext
            }
        }
        return nil
    }

    /// True when the text, taken whole, is a voice command — anchor plus
    /// phrase, or a bare command phrase standing as its own utterance.
    /// Echo suppression must not eat these.
    static func isCommandShaped(_ text: String) -> Bool {
        guard let match = VoiceCommand.consume(from: text, utterance: text) else { return false }
        return match.remaining.isEmpty
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
        // Words still moving means no pause yet — reschedule the watchdog.
        schedulePauseWatchdog()
        // Anchored commands fire on the partial, the moment the phrase is
        // complete — a barge-in cannot wait for the pause that finalizes the
        // utterance. The bare form needs a finalized utterance, so it is
        // checked only in commitFinal.
        pendingCommand = VoiceCommand.consume(from: currentText).map {
            PendingCommand(command: $0.command, consumedWords: $0.consumed)
        }
    }

    /// Arm the silence detector: in `pauseInterval`, if the hypothesis has
    /// not changed, commit it. Rescheduled by every partial change and torn
    /// down by every stop/restart, so it fires only on a genuine lull —
    /// exactly the pause the user places between their task and "run".
    private func schedulePauseWatchdog() {
        pauseWork?.cancel()
        let work = DispatchWorkItem { [weak self] in
            // We are on the main queue by construction, so the isolated
            // method is safe to reach synchronously.
            MainActor.assumeIsolated { self?.commitStalledPartial() }
        }
        pauseWork = work
        DispatchQueue.main.asyncAfter(deadline: .now() + Self.pauseInterval, execute: work)
    }

    private func cancelPauseWatchdog() {
        pauseWork?.cancel()
        pauseWork = nil
    }

    /// The lull happened: the hypothesis is stable, treat it as finalized.
    /// It rides the same commit path as a real final — cumulative-vs-append
    /// logic, command matching, once-only consumption — so a "run" lull
    /// fires the bare command and a mid-task lull simply lands the words.
    /// Internal so tests can drive the lull without waiting on the clock.
    /// (No isListening gate here: the watchdog is armed only from
    /// updatePartial and torn down by every stop and restart, so a stale
    /// fire cannot happen — and the guard would also lock tests out, since
    /// they drive the accumulator without a live session. A stale-fire
    /// commit is harmless anyway: it lands words that are already in the
    /// partial, and a recognizer final then replaces them by shape.)
    /// A recognizer final arriving on top is harmless too: the cumulative
    /// shape it then repeats replaces what we committed.
    func commitStalledPartial() {
        pauseWork = nil
        guard !partial.isEmpty else { return }
        commitFinal(partial)
    }

    /// Consume the pending command: strip its words out of the raw text so
    /// they neither ride into the sent message nor linger in the draft, and
    /// return the command for the composer to act on. Returns nil when
    /// nothing is pending — the once-only gate that keeps one spoken
    /// command from firing twice.
    @discardableResult
    func takeCommand() -> VoiceCommand? {
        guard let pending = pendingCommand else { return nil }
        pendingCommand = nil
        if let stripped = VoiceCommand.strippingLastWords(pending.consumedWords, from: currentText) {
            transcript = stripped
            partial = ""
        }
        return pending.command
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
        // The status read is synchronous; only the not-determined case
        // needs the async round-trip through tccd (and its on-screen prompt).
        // Keeping the common path — permission already decided — free of the
        // callback dance keeps start() honest about when it can block.
        let status = AVCaptureDevice.authorizationStatus(for: .audio)
        guard status == .notDetermined else { return status == .authorized }
        let granted = await AVCaptureDevice.requestAccess(for: .audio)
        return granted
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
