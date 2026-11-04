import SwiftUI

/// Colours are system colours, not brand colours.
///
/// The web UI's `#c96442` is a brand mark. A native app that hard-codes it
/// ignores the user's accent choice and looks foreign on a Mac, so the accent
/// here is `Color.accentColor` and everything else follows the system
/// appearance. Dark mode is not a toggle: it is whatever the system says.
enum Palette {
    static var surface: Color {
        Color(nsColor: .windowBackgroundColor)
    }

    static var raised: Color {
        Color(nsColor: .controlBackgroundColor)
    }

    static var userBubble: Color {
        Color.accentColor.opacity(0.12)
    }

    static var assistantBubble: Color {
        Color(nsColor: .controlBackgroundColor)
    }

    static var hairline: Color {
        Color(nsColor: .separatorColor)
    }

    static var secondaryText: Color {
        Color(nsColor: .secondaryLabelColor)
    }

    static var tertiaryText: Color {
        Color(nsColor: .tertiaryLabelColor)
    }

    static var danger: Color {
        Color(nsColor: .systemRed)
    }

    static var success: Color {
        Color(nsColor: .systemGreen)
    }
}
