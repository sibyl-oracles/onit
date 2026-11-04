import SwiftUI

/// Type scale and spacing, after Mercury Desktop (plan §5.6.3).
///
/// 13 pt body at 1.45 line height is the reading measure the plan settled on:
/// larger than macOS's 11 pt default because a transcript is read, not
/// scanned, and smaller than the web UI's because a native window is denser.
enum Typography {
    static let body = Font.system(size: 13)
    static let bodyEmphasis = Font.system(size: 13, weight: .medium)
    static let caption = Font.system(size: 11)
    static let title = Font.system(size: 15, weight: .semibold)
    static let mono = Font.system(size: 12, design: .monospaced)

    /// 1.45 line height, expressed as SwiftUI's multiplier on the font size.
    static let lineSpacing: CGFloat = 5.5
}

enum Metrics {
    static let sidebarWidth: CGFloat = 240
    static let railWidth: CGFloat = 64
    static let transcriptMaxWidth: CGFloat = 720
    static let controlRadius: CGFloat = 8
    static let cardRadius: CGFloat = 12
    static let gutter: CGFloat = 16
}

extension Animation {
    /// 120 ms ease-out, no spring. Mercury's motion is a settle, not a bounce.
    static let onit = Animation.easeOut(duration: 0.12)
}
