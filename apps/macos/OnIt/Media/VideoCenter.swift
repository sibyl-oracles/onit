import AVFoundation
import AVKit
import Foundation
import Observation

/// Playback state for one inline video attachment.
///
/// One player at a time, app-wide — the same rule AudioCenter applies to
/// audio. A second Play click anywhere — another session, another turn —
/// stops the first: two overlapping voices is never what a person wanted
/// from a chat transcript. Rows observe this center and render from it, so
/// a row does not own a player that outlives its message.
@MainActor
@Observable
final class VideoCenter {
    struct RowState: Equatable {
        /// The item decoded and the transport controls are live.
        var isReady = false
        var duration: TimeInterval = 0
        var loadFailed = false
    }

    /// The attachment currently loaded, by its server URL. Nil = nothing.
    private(set) var activeURL: String?
    /// The player the active row renders. Nil = nothing.
    private(set) var player: AVPlayer?

    private var states: [String: RowState] = [:]
    private var statusObservation: NSKeyValueObservation?
    private var tempURL: URL?

    /// Read-only access for rows; writes go through the center.
    func state(for url: String) -> RowState { states[url] ?? RowState() }

    /// Load an attachment's bytes and start playing them. Playing the same
    /// row twice is a no-op — the player is already on screen with its own
    /// transport controls, unlike audio where a second click means stop.
    func play(url: String, data: Data) {
        if activeURL == url { return }
        stopAll()

        // AVPlayer wants a file URL and the bytes arrive over the
        // connection. Keep the server-side extension as a demuxer hint.
        let ext = (url as NSString).pathExtension
        let name = "onit-video-\(UUID().uuidString)\(ext.isEmpty ? "" : ".\(ext)")"
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(name)
        do {
            try data.write(to: tmp, options: .atomic)
        } catch {
            states[url] = RowState(loadFailed: true)
            return
        }
        tempURL = tmp

        let item = AVPlayerItem(url: tmp)
        let player = AVPlayer(playerItem: item)
        self.player = player
        activeURL = url
        player.play()

        // Readiness arrives with the item's first load — unlike audio there
        // is no throwing constructor to catch a bad file synchronously. A
        // failed load is how a corrupt or unsupported file reports itself:
        // the row falls back to its download chip rather than spinning.
        statusObservation = item.observe(\.status, options: [.initial, .new]) {
            [weak self] item, _ in
            let ready = item.status == .readyToPlay
            let failed = item.status == .failed
            let duration = ready ? item.duration.seconds : 0
            Task { @MainActor in
                guard let self, self.activeURL == url else { return }
                if failed {
                    self.stopAll()
                    self.states[url] = RowState(loadFailed: true)
                } else if ready {
                    self.states[url] = RowState(isReady: true, duration: duration)
                }
            }
        }
    }

    func stopAll() {
        statusObservation?.invalidate()
        statusObservation = nil
        player?.pause()
        player = nil
        if let tempURL {
            try? FileManager.default.removeItem(at: tempURL)
            self.tempURL = nil
        }
        if let activeURL {
            states[activeURL] = RowState()
            self.activeURL = nil
        }
    }
}
