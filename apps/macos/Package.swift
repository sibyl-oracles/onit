// swift-tools-version: 6.0
import PackageDescription

// The macOS app is built as a Swift package and then wrapped into an .app
// bundle by Scripts/build.sh. A package rather than an .xcodeproj keeps the
// build reproducible from the command line and reviewable as text — the
// project file is the one artifact nobody can read in a diff.
let package = Package(
    name: "OnIt",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "OnIt",
            path: "OnIt",
            exclude: ["Resources"],
            swiftSettings: [
                .swiftLanguageMode(.v5),
            ]
        )
    ]
)
