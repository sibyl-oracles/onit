import Foundation

/// Locates the Python interpreter and the OnIt package the sidecar runs.
///
/// The first testable build runs the harness from the user's own Python rather
/// than a frozen PyInstaller binary. That is a deliberate trade: it makes the
/// app runnable in minutes instead of days, and it keeps the app honest about
/// which harness it is talking to. `ONIT_SIDECAR_PYTHON` and `ONIT_SIDECAR_ROOT`
/// override both, which is how a developer points the app at a working tree.
enum SidecarPaths {
    /// The interpreter that runs `python -m onit.cli`.
    ///
    /// The first candidate that can actually import the harness wins, not the
    /// first one that exists: `/usr/bin/python3` is always present and never
    /// has OnIt's dependencies, so picking it by existence alone would fail at
    /// startup with an import error instead of falling through.
    static func python() -> String? {
        if let override = ProcessInfo.processInfo.environment["ONIT_SIDECAR_PYTHON"],
           FileManager.default.isExecutableFile(atPath: override) {
            return override
        }
        let candidates = [
            "/opt/homebrew/bin/python3",
            "/opt/miniconda3/bin/python3",
            "/usr/local/bin/python3",
            "/usr/bin/python3",
        ].filter { FileManager.default.isExecutableFile(atPath: $0) }

        // Prefer an interpreter that can see the package we are about to run.
        if let root = packageRoot() {
            for candidate in candidates where canImport(candidate, root: root) {
                return candidate
            }
        }
        return candidates.first
    }

    /// Does this interpreter import `onit` with *root* on the path?
    private static func canImport(_ python: String, root: String) -> Bool {
        let probe = Process()
        probe.executableURL = URL(fileURLWithPath: python)
        probe.arguments = ["-c", "import onit"]
        var env = ProcessInfo.processInfo.environment
        env["PYTHONPATH"] = root
        probe.environment = env
        probe.standardOutput = FileHandle.nullDevice
        probe.standardError = FileHandle.nullDevice
        do {
            try probe.run()
        } catch {
            return false
        }
        probe.waitUntilExit()
        return probe.terminationStatus == 0
    }

    /// The directory that must be on `PYTHONPATH` for `onit` to import.
    ///
    /// Resolution order, first hit wins:
    /// 1. `ONIT_SIDECAR_ROOT` — how a developer points the app at a tree.
    /// 2. `onit-src/` inside the bundle — how a shipped app carries its own.
    /// 3. The working tree this app was built from, recorded at build time in
    ///    `OnItBuildRoot` and found by walking up from the executable.
    static func packageRoot() -> String? {
        if let override = ProcessInfo.processInfo.environment["ONIT_SIDECAR_ROOT"],
           FileManager.default.fileExists(atPath: override + "/onit/__init__.py") {
            return override
        }
        if let bundled = Bundle.main.resourceURL?.appending(path: "onit-src").path,
           FileManager.default.fileExists(atPath: bundled + "/onit/__init__.py") {
            return bundled
        }
        // Walk up from the executable looking for a sibling `onit/` package.
        // build/OnIt.app/Contents/MacOS/OnIt -> apps/macos/build/... -> repo.
        var directory = URL(fileURLWithPath: CommandLine.arguments[0])
            .resolvingSymlinksInPath()
            .deletingLastPathComponent()
        for _ in 0..<8 {
            if FileManager.default.fileExists(
                atPath: directory.appending(path: "onit/__init__.py").path
            ) {
                return directory.path
            }
            directory = directory.deletingLastPathComponent()
        }
        return nil
    }

    /// Where the app keeps its own state, separate from `~/.onit`.
    static func applicationSupport() -> URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        let dir = base.appending(path: "OnIt")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }
}

/// Owns the local harness process.
///
/// One sidecar per app, shared by every session: sessions are a server-side
/// concept and the backend already runs each one concurrently. Spawning a
/// process per session would multiply the model connections for no gain.
@MainActor
@Observable
final class SidecarManager {
    enum State: Equatable {
        case idle
        case starting
        case ready(port: Int)
        case failed(String)
        case stopped

        var isReady: Bool { if case .ready = self { return true }; return false }
        var port: Int? { if case let .ready(port) = self { return port }; return nil }
    }

    private(set) var state: State = .idle
    private(set) var logTail: [String] = []
    /// The joined form of the last N log lines, recomputed only when
    /// `logTail` changes. Three views render this; joining 200 lines per
    /// stderr chunk in each of them is work nobody asked for.
    private(set) var logText: String = ""

    /// How much of the tail the views show, so the cache matches what is
    /// rendered rather than joining all 200 lines every time.
    static let displayedLogLines = 60

    private var process: Process?
    private var outputPipe: Pipe?
    /// True while `stop()` is tearing the harness down on purpose. The exit
    /// handler must not report a requested stop as a crash.
    private var isStopping = false
    /// Called when the harness exits without being asked to. `AppModel` uses
    /// it to drop the local connection offline and offer a restart.
    var onDeath: (() -> Void)?

    /// Start the harness on a free port and wait until `/api/config` answers.
    ///
    /// The port is chosen by the OS (bind :0, read, close) rather than fixed,
    /// so two copies of the app — or the browser UI on 9000 — never collide.
    func start(port explicitPort: Int? = nil) async {
        guard process == nil else { return }
        isStopping = false
        state = .starting
        logTail.removeAll()
        Self.trace("start: python=\(SidecarPaths.python() ?? "nil") root=\(SidecarPaths.packageRoot() ?? "nil")")

        guard let python = SidecarPaths.python() else {
            state = .failed("No Python 3 found. Set ONIT_SIDECAR_PYTHON.")
            return
        }
        guard let root = SidecarPaths.packageRoot() else {
            state = .failed(
                "OnIt package not found. Set ONIT_SIDECAR_ROOT to the repo root."
            )
            return
        }

        let port = explicitPort ?? PortAllocator.free()
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: python)
        proc.arguments = ["-m", "onit.cli", "serve", "web", "--port", "\(port)", "--no-login"]
        proc.currentDirectoryURL = URL(fileURLWithPath: root)

        var env = ProcessInfo.processInfo.environment
        env["PYTHONPATH"] = root
        env["PYTHONUNBUFFERED"] = "1"
        // The app is a single-user desktop client; a login wall in front of a
        // loopback port would only be a second password to type.
        env["ONIT_NO_LOGIN"] = "1"
        proc.environment = env

        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            Task { @MainActor [weak self] in self?.appendLog(text) }
        }

        do {
            try proc.run()
            Self.trace("start: spawned pid \(proc.processIdentifier) on port \(port)")
        } catch {
            state = .failed("Could not start the harness: \(error.localizedDescription)")
            return
        }
        process = proc
        outputPipe = pipe

        // A crash after startup must be visible. Without this the rail keeps
        // a healthy-looking dot while every request fails with connection
        // errors, and nothing tells the user the harness is gone.
        proc.terminationHandler = { [weak self] exited in
            Task { @MainActor [weak self] in
                self?.harnessDied(exited)
            }
        }

        // Poll rather than parse stdout: the harness prints a human banner and
        // nothing machine-readable yet (plan §11 item 2). Polling `/api/config`
        // is the contract that already exists.
        let base = URL(string: "http://127.0.0.1:\(port)")!
        let client = OnItClient(baseURL: base)
        for _ in 0..<120 {
            if proc.isRunning == false {
                state = .failed("The harness exited during startup.")
                return
            }
            if (try? await client.config()) != nil {
                state = .ready(port: port)
                Self.trace("start: ready on \(port)")
                return
            }
            try? await Task.sleep(for: .milliseconds(250))
        }
        state = .failed("The harness did not answer on port \(port) within 30 s.")
    }

    func stop() {
        isStopping = true
        outputPipe?.fileHandleForReading.readabilityHandler = nil
        if let process, process.isRunning {
            process.terminate()
            // The grace wait happens off the main actor: a synchronous wait
            // here would freeze the window for up to three seconds while the
            // harness closes its sockets.
            let proc = process
            DispatchQueue.global(qos: .userInitiated).async {
                let deadline = Date().addingTimeInterval(3)
                while proc.isRunning && Date() < deadline {
                    usleep(50_000)
                }
                if proc.isRunning { kill(proc.processIdentifier, SIGKILL) }
            }
        }
        process = nil
        outputPipe = nil
        state = .stopped
    }

    /// The harness process ended. Distinguish "we stopped it" from "it died".
    private func harnessDied(_ exited: Process) {
        guard !isStopping, exited === process else { return }
        process = nil
        outputPipe = nil
        Self.trace("died: status \(exited.terminationStatus)")
        state = .failed("The harness exited unexpectedly (status \(exited.terminationStatus)).")
        onDeath?()
    }

    private func appendLog(_ text: String) {
        for line in text.split(separator: "\n", omittingEmptySubsequences: true) {
            logTail.append(String(line))
        }
        if logTail.count > 200 {
            logTail.removeFirst(logTail.count - 200)
        }
        logText = logTail.suffix(Self.displayedLogLines).joined(separator: "\n")
    }

    /// Write a line to stderr, which the build script and `log stream` both see.
    ///
    /// Startup failures happen before any window exists, so a trace is the only
    /// way to tell "the harness never spawned" from "it spawned and died".
    nonisolated static func trace(_ message: String) {
        FileHandle.standardError.write(Data("[onit-app] \(message)\n".utf8))
    }
}

/// Ask the kernel for an unused port.
///
/// Bind to :0, read back what it assigned, close. There is a race between the
/// close and the child's bind, but it is a narrow one on loopback and the
/// alternative — a fixed port — fails far more often in practice.
enum PortAllocator {
    static func free() -> Int {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { return 9000 }
        defer { close(fd) }

        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_addr.s_addr = inet_addr("127.0.0.1")
        addr.sin_port = 0

        let bound = withUnsafePointer(to: &addr) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bound == 0 else { return 9000 }

        var result = sockaddr_in()
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        let named = withUnsafeMutablePointer(to: &result) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                getsockname(fd, $0, &length)
            }
        }
        guard named == 0 else { return 9000 }
        return Int(UInt16(bigEndian: result.sin_port))
    }
}
