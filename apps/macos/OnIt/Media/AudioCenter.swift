import AVFoundation
import Foundation
import Observation

/// Playback state for one inline audio attachment.
///
/// One player at a time, app-wide. A second Play click anywhere — another
/// session, another turn — stops the first: two overlapping voices is never
/// what a person wanted from a chat transcript. Rows observe this center and
/// render from it, so a row does not own a player that outlives its message.
@MainActor
@Observable
final class AudioCenter: NSObject, AVAudioPlayerDelegate {
    struct RowState: Equatable {
        var isPlaying = false
        var elapsed: TimeInterval = 0
        var duration: TimeInterval = 0
        var loadFailed = false
    }

    /// The attachment currently loaded, by its server URL. Nil = nothing.
    private(set) var activeURL: String?
    private var player: AVAudioPlayer?
    private var tick: Timer?

    private var states: [String: RowState] = [:]

    /// Read-only access for rows; writes go through the center.
    func state(for url: String) -> RowState { states[url] ?? RowState() }

    func toggle(url: String, data: Data) {
        if activeURL == url {
            stopAll()
            return
        }
        stopAll()
        do {
            let player = try AVAudioPlayer(data: data)
            player.delegate = self
            guard player.play() else {
                states[url] = RowState(loadFailed: true)
                return
            }
            self.player = player
            activeURL = url
            states[url] = RowState(isPlaying: true, elapsed: 0, duration: player.duration)
            let timer = Timer(timeInterval: 0.25, repeats: true) { [weak self] _ in
                Task { @MainActor in self?.tickOnce() }
            }
            RunLoop.main.add(timer, forMode: .common)
            self.tick = timer
        } catch {
            states[url] = RowState(loadFailed: true)
        }
    }

    private func tickOnce() {
        guard let player, let activeURL else { return }
        states[activeURL] = RowState(
            isPlaying: true,
            elapsed: player.currentTime,
            duration: player.duration
        )
    }

    func stopAll() {
        tick?.invalidate()
        tick = nil
        player?.stop()
        player = nil
        if let activeURL {
            states[activeURL] = RowState()
            self.activeURL = nil
        }
    }

    // MARK: AVAudioPlayerDelegate

    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor in self.stopAll() }
    }
}
