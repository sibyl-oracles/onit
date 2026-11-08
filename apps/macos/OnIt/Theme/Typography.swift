import SwiftUI

/// Type scale and spacing, after Mercury Desktop (plan §5.6.3).
///
/// 13 pt body at 1.45 line height is the reading measure the plan settled on:
/// larger than macOS's 11 pt default because a transcript is read, not
/// scanned, and smaller than the web UI's because a native window is denser.
///
/// The scale is a global multiplier, not an environment value, because the
/// views that need it sit behind `.equatable()` — an environment change would
/// not be seen by a view whose body is skipped. The multiplier travels as a
/// plain parameter instead (see MessageBubble), and the app-level source of
/// truth is AppModel.textScale, which persists it and writes it here.
@MainActor
enum Typography {
    /// The user's text size choice, 1.0 by default. Written only by
    /// AppModel.textScale; read by every font below and passed down as a
    /// parameter where equality-gated views need it.
    static var scale: CGFloat = 1.0

    private static func sized(_ size: CGFloat) -> Font {
        .system(size: (size * scale).rounded(.toNearestOrEven))
    }

    static var body: Font { sized(13) }
    static var bodyEmphasis: Font { .system(size: (13 * scale).rounded(), weight: .medium) }
    static var caption: Font { sized(11) }
    static var title: Font { .system(size: (15 * scale).rounded(), weight: .semibold) }
    static var mono: Font { .system(size: (12 * scale).rounded(), design: .monospaced) }

    /// 1.45 line height, expressed as SwiftUI's multiplier on the font size.
    /// Scales with the type so the measure stays a measure.
    static var lineSpacing: CGFloat { 5.5 * scale }
}

enum Metrics {
    static let sidebarWidth: CGFloat = 240
    static let railWidth: CGFloat = 64
    /// The transcript column's widest row. A hard ceiling, not the width: the
    /// actual column grows with the window up to this, then holds (ChatView).
    static let transcriptMaxWidth: CGFloat = 1100
    /// Below this window width the column stops shrinking, so text never
    /// wraps against the scrollbar.
    static let transcriptMinWidth: CGFloat = 520
    static let controlRadius: CGFloat = 8
    static let cardRadius: CGFloat = 12
    static let gutter: CGFloat = 16
}

extension Animation {
    /// 120 ms ease-out, no spring. Mercury's motion is a settle, not a bounce.
    static let onit = Animation.easeOut(duration: 0.12)
}
