# Q-Volta theme for the vendored llama.cpp web UI

The built-in chat UI is the llama.cpp web UI (MIT, `../webui/LICENSE`) rebuilt from source with this theme. `scripts/vendor_webui.sh` copies the upstream `ui/` directory, applies the theme, runs `npm ci && npm run build` and gzips the result into `../webui/gzip/`.

* `q-volta-theme.patch` (`patch -p1`, from the `ui/` root): edits to existing upstream files
  * `lib/ui/.../styles/theme.css`: drops the bundled Llama fonts (system font stack instead)
  * `apps/chat/src/app.css`: imports `q-theme.css`
  * `lib/ui/.../misc/Logo.svelte`: the Q mark instead of the llama mark (used by the sidebar key, model lists, greeting)
  * `apps/chat/scripts/brand-favicons.ts`: favicons come from `scripts/q-brand/` (the PWA icons and splash screens are generated from them)
  * `app.constants.ts`, `pwa.constants.ts`, `search/+page.svelte`, `ServerErrorSplash`, `McpServerForm`, `ChatFormActionAddToolsSubmenu`: title/name/strings (`Q-Volta`, `qv_server`), manifest colours
  * `settings.constants.ts`, `+layout.svelte`: dark is the default theme (a stored user choice still wins)
  * `ChatScreenGreeting.svelte`: wordmark and chips (model alias, context, vision/audio, all from `/props`); `SidebarNavigation.svelte`: wordmark; `SettingsChat.svelte`, `settings/index.ts`: the credit block; `types/api.d.ts`: `model_alias`
* `files/`: new files copied over the tree: `apps/chat/src/q-theme.css` (all colour tokens, light and dark, WCAG-AA checked, the gradient primary button, wordmark/chip styles), `apps/chat/scripts/q-brand/favicon*.svg` (Q mark), `QAbout.svelte` (credit with link to the licence text).

Updating the UI upstream: check out the new llama.cpp commit, run `scripts/vendor_webui.sh <checkout>`; if `patch` rejects a hunk, fix it in a scratch copy of `ui/` (`git init` it, apply, resolve, `git diff` the tracked paths back into `q-volta-theme.patch`). Because the UI is built on semantic design tokens, most upstream changes only touch `q-theme.css` values, not the patch.
