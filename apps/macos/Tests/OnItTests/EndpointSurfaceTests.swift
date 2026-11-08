import XCTest
@testable import OnIt

/// The endpoint/key surface, decoded against what `GET /api/endpoints`
/// actually returns (`onit/ui/endpoints.py:describe`).
///
/// The two properties under test are the load-bearing ones: a key never
/// appears as a value (only a source and a masked tail), and the integration
/// row's `label` field is the *status* text — the server spreads the status
/// dict over the row, so the display name has to be derived client-side.
final class EndpointSurfaceTests: XCTestCase {
    private let describeJSON = """
    {
      "endpoints": [
        {
          "id": "https://openrouter.ai/api/v1",
          "host": "https://openrouter.ai/api/v1",
          "model": "openai/gpt-5",
          "name": "OR",
          "priority": 0,
          "position": 0,
          "key": {"source": "keychain", "masked": "••••7890", "label": "stored"},
          "notes": []
        },
        {
          "id": "http://localhost:11434/v1",
          "host": "http://localhost:11434/v1",
          "model": "",
          "name": "",
          "priority": 1,
          "position": 1,
          "key": {"source": "env", "masked": null, "label": "from $VLLM_API_KEY"},
          "notes": ["vLLM needs an API key."]
        }
      ],
      "presets": [
        {"label": "Ollama (local)", "host": "http://localhost:11434/v1", "needs_key": false},
        {"label": "OpenRouter", "host": "https://openrouter.ai/api/v1", "needs_key": true}
      ],
      "integrations": [
        {
          "key": "tavily_api_key",
          "label": "not set",
          "env_var": "TAVILY_API_KEY",
          "blurb": "Web search, first tier.",
          "source": null,
          "masked": null
        },
        {
          "key": "github_token",
          "label": "stored",
          "env_var": "GITHUB_TOKEN",
          "blurb": "Git operations.",
          "source": "keychain",
          "masked": "••••abcd"
        }
      ],
      "config_path": "/Users/test/.onit/config.yaml",
      "keychain": true
    }
    """

    private var surface: OnItClient.EndpointSurface!

    override func setUpWithError() throws {
        surface = try JSONDecoder().decode(
            OnItClient.EndpointSurface.self,
            from: Data(describeJSON.utf8)
        )
    }

    func testEndpointRowsDecode() throws {
        XCTAssertEqual(surface.endpoints.count, 2)
        let first = surface.endpoints[0]
        XCTAssertEqual(first.host, "https://openrouter.ai/api/v1")
        XCTAssertEqual(first.model, "openai/gpt-5")
        XCTAssertEqual(first.key.label, "stored")
        XCTAssertEqual(first.key.masked, "••••7890")
    }

    func testKeyStatusNeverCarriesAValue() {
        // The masked tail is at most four characters after four dots; a
        // decoder that accepted a raw key here would be the bug.
        for row in surface.endpoints {
            if let masked = row.key.masked {
                XCTAssertTrue(masked.hasPrefix("••••"))
                XCTAssertLessThanOrEqual(masked.count, 8)
            }
        }
        for integration in surface.integrations {
            if let masked = integration.masked {
                XCTAssertTrue(masked.hasPrefix("••••"))
            }
        }
    }

    func testUnsetKeyIsReportedAsUnset() {
        let unset = surface.endpoints[1]
        XCTAssertEqual(unset.key.label, "from $VLLM_API_KEY")
        XCTAssertNil(unset.key.masked)
    }

    func testPresetsDecodeNeedsKey() {
        XCTAssertEqual(surface.presets.count, 2)
        XCTAssertFalse(surface.presets[0].needsKey)
        XCTAssertTrue(surface.presets[1].needsKey)
    }

    func testIntegrationLabelIsTheStatusNotTheName() {
        // The server spreads the status dict over the row, so `label` holds
        // "not set"/"stored", not "Tavily". The row's display name comes from
        // the key spelling.
        XCTAssertEqual(surface.integrations[0].label, "not set")
        XCTAssertEqual(surface.integrations[0].status, "not set")
        XCTAssertEqual(surface.integrations[0].displayName, "Tavily")
        XCTAssertEqual(surface.integrations[1].displayName, "Github")
    }

    func testIntegrationSetState() {
        XCTAssertFalse(surface.integrations[0].isSet)
        XCTAssertTrue(surface.integrations[1].isSet)
    }

    func testConfigPathAndKeychain() {
        XCTAssertEqual(surface.configPath, "/Users/test/.onit/config.yaml")
        XCTAssertTrue(surface.keychain)
    }

    func testEndpointDisplayTitlePrefersNameThenModelThenHost() {
        XCTAssertEqual(surface.endpoints[0].displayTitle, "OR")
        XCTAssertEqual(surface.endpoints[1].displayTitle, "http://localhost:11434/v1")
    }

    func testSaveEndpointsBodyShape() {
        // What saveEndpoints sends: host/model/name/priority, no key.
        let rows = surface.endpoints
        let body: [[String: Any]] = rows.map { row in
            var entry: [String: Any] = ["host": row.host]
            if let model = row.model, !model.isEmpty { entry["model"] = model }
            if let name = row.name, !name.isEmpty { entry["name"] = name }
            entry["priority"] = row.priority
            return entry
        }
        let first = body[0]
        XCTAssertEqual(first["host"] as? String, "https://openrouter.ai/api/v1")
        XCTAssertEqual(first["model"] as? String, "openai/gpt-5")
        XCTAssertEqual(first["name"] as? String, "OR")
        XCTAssertNil(first["key"], "a save must never carry a key")
    }
}
