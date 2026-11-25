package com.toni.wwe13;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.ClipData;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.res.AssetManager;
import android.database.Cursor;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.graphics.drawable.StateListDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.ParcelFileDescriptor;
import android.provider.OpenableColumns;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.TextView;

import java.io.File;
import java.io.FileInputStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * WWE '13 launcher: game file setup (disc image, title update, DLC), settings, custom music, creations and
 * save backups. "Start the game directly" skips it; the icon's long-press "Settings" shortcut always opens it.
 */
public class LauncherActivity extends Activity {
    private static final int REQ_ISO = 1;
    private static final int REQ_PACKAGE = 2;
    private static final int REQ_SONGS = 3;
    private static final int REQ_CREATIONS = 4;
    private static final int REQ_MOD = 5;
    private static final int REQ_SYNC_NEW = 6;
    private static final int REQ_SYNC_OPEN = 7;

    private static final String[] RES_LABELS = {"480p (fastest)", "576p", "720p (original)", "1080p", "1440p (sharpest)"};
    private static final String[] FPS_LABELS = {"60 FPS, entrances at 30 (like the original)", "60 FPS everywhere (recommended)", "Lock matches to 30 FPS (cooler, longer battery)"};
    private static final String BUNDLED_DRIVER = "Turnip_T30";
    private static final int REQ_DRIVER = 8;
    private static final int REQ_K14 = 9;
    private static final String K14_MOD = "WWE 2K14 superstars";
    private static final String[] LANG_LABELS = {"English", "German", "French", "Spanish", "Italian"};
    private static final int[] LANG_IDS = {1, 3, 4, 5, 6};

    // WWE '13 look: black, the game's red, white bold capitals.
    private static final int BG = 0xFF070707;
    private static final int CARD = 0xE6141414;
    private static final int ACCENT = 0xFFD7191F;
    private static final int TEXT = 0xFFF2F2F2;
    private static final int DIM = 0xFF9A9AA6;

    private SharedPreferences prefs;
    private File root;
    private final Handler handler = new Handler(Looper.getMainLooper());

    private TextView statusText;
    private Button playButton;
    private LinearLayout songList;
    private LinearLayout creationList;
    private LinearLayout backupList;
    private LinearLayout modList;
    private TextView syncStatus;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        prefs = getSharedPreferences("launcher", MODE_PRIVATE);
        root = getExternalFilesDir(null);
        LauncherNative.init(root.getAbsolutePath());
        installBundledFiles();

        boolean openedAsSettings = getIntent() != null && getIntent().getComponent() != null
                && getIntent().getComponent().getClassName().endsWith("SettingsAlias");
        if (!openedAsSettings && prefs.getBoolean("skip_launcher", false) && gameReady()) {
            if (Features.EXTRAS && SaveSync.isSetUp(this) && prefs.getBoolean("sync_auto", true)) {
                // Quick sync first; a conflict is left for the launcher screen.
                setContentView(text("  Syncing saves…", 18, TEXT));
                new Thread(() -> {
                    SaveSync.Result r = SaveSync.sync(this, null, false);
                    handler.post(() -> {
                        recordSync(r);
                        if (r.outcome == SaveSync.Outcome.CONFLICT) buildUi(); else startSoon();
                    });
                }).start();
                return;
            }
            startSoon();
            return;
        }
        buildUi();
    }

    // "Start the game directly": a short screen first, where any button or a tap opens the settings instead.
    private boolean startPending;
    private final Runnable startRunnable = () -> { if (startPending) { startPending = false; startGame(); } };

    private void startSoon() {
        TextView t = text("WWE'13\n\nPRESS ANY BUTTON OR TAP FOR SETTINGS", 18, TEXT);
        t.setGravity(Gravity.CENTER);
        t.setTypeface(Typeface.create("sans-serif-condensed", Typeface.BOLD_ITALIC));
        t.setBackground(new XBackground());
        t.setOnClickListener(v -> openSettingsInstead());
        setContentView(t);
        startPending = true;
        handler.postDelayed(startRunnable, 2000);
    }

    private void openSettingsInstead() {
        if (!startPending) return;
        startPending = false;
        handler.removeCallbacks(startRunnable);
        buildUi();
    }

    @Override
    public boolean onKeyDown(int keyCode, android.view.KeyEvent event) {
        if (startPending) {
            openSettingsInstead();
            return true;
        }
        return super.onKeyDown(keyCode, event);
    }

    // ------------------------------------------------------------------------------------------ first run files

    private void installBundledFiles() {
        int bundled = 3;  // bump when the bundled files change
        if (prefs.getInt("bundled_version", 0) >= bundled) return;
        try {
            // Turnip 26.3.0-R3 (bundled before) draws broken geometry in matches; T30 draws correctly.
            deleteTree(new File(getFilesDir(), "drivers/Turnip_v26.3.0-R3"));
            File drivers = new File(getFilesDir(), "drivers/" + BUNDLED_DRIVER);
            drivers.mkdirs();
            copyAsset("drivers/libvulkan_freedreno.bin", new File(drivers, "libvulkan_freedreno.so"));
            File shaders = new File(root, "shader-cache");
            shaders.mkdirs();
            AssetManager am = getAssets();
            String[] names = am.list("shader-cache");
            if (names != null) {
                for (String n : names) {
                    File dest = new File(shaders, n);
                    // Keep a cache the game has already grown while playing.
                    if (!dest.exists() || dest.length() == 0) copyAsset("shader-cache/" + n, dest);
                }
            }
            prefs.edit().putInt("bundled_version", bundled).apply();
        } catch (Exception e) {
            // The game still runs with the phone's driver and builds its shader cache while playing.
        }
    }

    private void copyAsset(String asset, File dest) throws Exception {
        try (InputStream in = getAssets().open(asset); OutputStream out = new FileOutputStream(dest)) {
            copy(in, out);
        }
    }

    private static void copy(InputStream in, OutputStream out) throws Exception {
        byte[] buf = new byte[1 << 20];
        int n;
        while ((n = in.read(buf)) > 0) out.write(buf, 0, n);
    }

    // ------------------------------------------------------------------------------------------ UI

    private void buildUi() {
        tabs.clear();
        tabButtons.clear();
        // Two panes like the game's menus: the menu on the left, the chosen page on the right, over a red X.
        LinearLayout rootView = new LinearLayout(this);
        rootView.setOrientation(LinearLayout.HORIZONTAL);
        rootView.setBackground(new XBackground());

        LinearLayout nav = new LinearLayout(this);
        nav.setOrientation(LinearLayout.VERTICAL);
        nav.setPadding(dp(28), dp(22), dp(12), dp(16));
        TextView logo = text("WWE'13", 44, TEXT);
        logo.setTypeface(Typeface.create("sans-serif-condensed", Typeface.BOLD_ITALIC));
        logo.setShadowLayer(dp(6), 0, 0, ACCENT);
        nav.addView(logo);
        View bar = new View(this);
        bar.setBackgroundColor(ACCENT);
        nav.addView(bar, new LinearLayout.LayoutParams(dp(170), dp(4)));
        TextView ver = text("ANDROID PORT  ·  v" + Updater.currentVersion(this), 11, DIM);
        ver.setLetterSpacing(0.08f);
        nav.addView(ver);
        navItems = new LinearLayout(this);
        navItems.setOrientation(LinearLayout.VERTICAL);
        navItems.setPadding(0, dp(14), 0, 0);
        nav.addView(navItems);
        ScrollView navScroll = new ScrollView(this);
        navScroll.addView(nav);
        rootView.addView(navScroll, new LinearLayout.LayoutParams(dp(250), ViewGroup.LayoutParams.MATCH_PARENT));

        ScrollView scroll = new ScrollView(this);
        contentScroll = scroll;
        LinearLayout page = new LinearLayout(this);
        page.setOrientation(LinearLayout.VERTICAL);
        page.setPadding(dp(12), dp(22), dp(28), dp(28));
        scroll.addView(page);
        rootView.addView(scroll, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1f));

        // ---- PLAY
        currentTab = "PLAY";
        // Update banner (filled in when GitHub has a newer release).
        updateBox = new LinearLayout(this);
        updateBox.setOrientation(LinearLayout.VERTICAL);
        page.addView(updateBox);
        tag(updateBox);
        if (prefs.getBoolean("check_updates", true)) Updater.check(this, true, this::showUpdate);

        playButton = button("PLAY", v -> startGame());
        playButton.setTextSize(TypedValue.COMPLEX_UNIT_SP, 30);
        playButton.setTypeface(Typeface.create("sans-serif-condensed", Typeface.BOLD_ITALIC));
        playButton.setPadding(dp(24), dp(18), dp(24), dp(18));
        StateListDrawable playBg = new StateListDrawable();
        playBg.addState(new int[]{android.R.attr.state_focused}, rounded(0xFFF0262C));
        playBg.addState(new int[]{android.R.attr.state_pressed}, rounded(0xFF9E1015));
        playBg.addState(new int[]{}, rounded(ACCENT));
        playButton.setBackground(playBg);
        page.addView(playButton, matchWidth(dp(6)));
        tag(playButton);
        CheckBox skip = checkbox("Start the game directly next time (press any button while it starts to come back here)",
                prefs.getBoolean("skip_launcher", false), c -> prefs.edit().putBoolean("skip_launcher", c).apply());
        page.addView(skip);
        tag(skip);
        addCrashNotice(page);

        // Game files
        LinearLayout files = section(page, "Game files");
        statusText = text("", 15, TEXT);
        files.addView(statusText);
        LinearLayout fileButtons = row();
        fileButtons.addView(button("Install from my disc image (.iso)…", v -> pick(REQ_ISO, false)));
        fileButtons.addView(button("Add title update or DLC…", v -> pick(REQ_PACKAGE, true)));
        files.addView(fileButtons);
        files.addView(text("Use your own WWE '13 Xbox 360 disc image and title update 2.0.1.0. The game is copied into the app, "
                + "so you can delete the image afterwards.", 13, DIM));

        // ---- GRAPHICS
        currentTab = "GRAPHICS";
        LinearLayout settings = section(page, "Graphics");
        settings.addView(spinnerRow("Resolution", RES_LABELS, prefs.getInt("resolution", 2),
                i -> prefs.edit().putInt("resolution", i).apply()));
        settings.addView(spinnerRow("Frame rate", FPS_LABELS, prefs.getInt("frame_rate", 1),
                i -> prefs.edit().putInt("frame_rate", i).apply()));
        final List<String> driverIds = driverIds();
        List<String> driverLabels = new ArrayList<>();
        for (String id : driverIds) driverLabels.add(driverLabel(id));
        settings.addView(spinnerRow("Graphics driver", driverLabels.toArray(new String[0]),
                Math.max(0, driverIds.indexOf(currentDriver())),
                i -> prefs.edit().putString("driver_name", driverIds.get(i)).apply()));
        settings.addView(button("Add GPU driver (.zip)…", v -> pick(REQ_DRIVER, false)), wrap(dp(4)));
        settings.addView(checkbox("Faster anti-aliasing (2x instead of 4x)", prefs.getBoolean("aa_faster", false),
                c -> prefs.edit().putBoolean("aa_faster", c).apply()));
        settings.addView(checkbox("Stretch the picture to fill the screen", prefs.getBoolean("stretch", false),
                c -> prefs.edit().putBoolean("stretch", c).apply()));
        settings.addView(checkbox("Show an FPS counter in the game", prefs.getBoolean("show_fps", false),
                c -> prefs.edit().putBoolean("show_fps", c).apply()));
        settings.addView(text("The first start prepares the game's graphics shaders (about 25 seconds on a black screen) so "
                + "menus and matches don't stutter. Later starts reuse them and take about a second.", 13, DIM));

        // ---- GAME
        currentTab = "GAME";
        settings = section(page, "Game");
        int langIndex = 0;
        int lang = LauncherNative.getLanguage();
        for (int i = 0; i < LANG_IDS.length; ++i) if (LANG_IDS[i] == lang) langIndex = i;
        settings.addView(spinnerRow("Game language", LANG_LABELS, langIndex, i -> {
            String err = LauncherNative.setLanguage(LANG_IDS[i]);
            if (!err.isEmpty()) toastDialog(err);
        }));
        settings.addView(checkbox("Unlock all superstars, arenas and attires (your save is not changed)", prefs.getBoolean("unlock_all", false),
                c -> prefs.edit().putBoolean("unlock_all", c).apply()));
        if (Features.EXTRAS)
            settings.addView(checkbox("Training mode: the CPU never reverses (plays on a copy of your save, your real save is not changed)",
                    prefs.getBoolean("training", false), c -> prefs.edit().putBoolean("training", c).apply()));
        settings.addView(spinnerRow("CPU kick-outs", new String[] {"Normal (like the original)", "Fewer after big moves",
                        "Much fewer after big moves and finishers"},
                prefs.getInt("cpu_kickouts", 0), i -> prefs.edit().putInt("cpu_kickouts", i).apply()));
        settings.addView(text("Only changes how often the CPU kicks out of your pins after heavy damage or a finisher. "
                + "Difficulty, reversals and your own kick-outs stay the same.", 13, DIM));
        settings.addView(spinnerRow("Button icons in the game", new String[] {"Xbox (A B X Y)", "PlayStation (✕ ○ □ △)"},
                prefs.getInt("button_icons", 0), i -> prefs.edit().putInt("button_icons", i).apply()));
        // ---- CONTROLS
        currentTab = "CONTROLS";
        settings = section(page, "Touch controls");
        settings.addView(spinnerRow("Touch controls", new String[] {"Automatic (hidden while a controller is connected)", "Always on", "Off"},
                prefs.getInt("touch_mode", 0), i -> prefs.edit().putInt("touch_mode", i).apply()));
        settings.addView(spinnerRow("Touch controls opacity", new String[] {"Light (30%)", "Medium (50%)", "Strong (75%)"},
                prefs.getInt("touch_opacity", 1), i -> prefs.edit().putInt("touch_opacity", i).apply()));
        settings.addView(spinnerRow("Touch button size", new String[] {"Small", "Medium", "Large"},
                prefs.getInt("touch_size", 1), i -> prefs.edit().putInt("touch_size", i).apply()));
        settings.addView(button("Edit touch layout…", v -> startActivity(new Intent(this, TouchLayoutActivity.class))), wrap(dp(4)));
        settings.addView(checkbox("Vibrate when a touch button is pressed", prefs.getBoolean("touch_vibrate", true),
                c -> prefs.edit().putBoolean("touch_vibrate", c).apply()));
        settings.addView(text("Touch controls: tap RUN to switch running on or off (or push the stick past its ring). HIDE / SHOW "
                + "in the game's top-right corner hides the buttons and brings them back; hold it (or use Edit touch layout) to move and resize them.", 13, DIM));

        // ---- MUSIC
        currentTab = "MUSIC";
        LinearLayout music = section(page, "Custom music (entrance music)");
        music.addView(text("Songs you add here can be used as entrance music: Creations → Entrance → Finalize → Music → "
                + "User Playlist. MP3, FLAC, OGG and WAV.", 13, DIM));
        music.addView(button("Add songs…", v -> pick(REQ_SONGS, true)), wrap(dp(8)));
        songList = new LinearLayout(this);
        songList.setOrientation(LinearLayout.VERTICAL);
        music.addView(songList);

        currentTab = "MODS";
        if (Features.EXTRAS) {
            // Creations (mods)
            LinearLayout mods = section(page, "Creations and mods");
            mods.addView(text("Install community creations: Superstars (CAWs), entrances, arenas and logos "
                    + "(Xbox 360 creation packages). Your saves are backed up first.", 13, DIM));
            mods.addView(button("Import creations…", v -> pick(REQ_CREATIONS, true)), wrap(dp(8)));
            creationList = new LinearLayout(this);
            creationList.setOrientation(LinearLayout.VERTICAL);
            mods.addView(creationList);

            // Superstars from WWE 2K14, built on this device from the player's own disc
            LinearLayout k14 = section(page, "Superstars from WWE 2K14");
            k14.addView(text("Have WWE 2K14 for Xbox 360? Pick your own disc image (.iso) and the 38 superstars WWE '13 "
                    + "doesn't have are added as a mod, with their models, names, pictures, titantrons, entrance music, "
                    + "entrances, movesets and ratings. Nothing is downloaded: everything comes from your disc. It takes "
                    + "a few minutes and about 1 GB of space. In the game, press R3 on the superstar select screen to "
                    + "see them.", 13, DIM));
            k14.addView(button("Port from my WWE 2K14 disc (.iso)…", v -> pick(REQ_K14, false)), wrap(dp(8)));

            // Game file mods
            LinearLayout gameMods = section(page, "Game file mods");
            gameMods.addView(text("Mods made for modded Xbox 360 consoles that replace game files: wrestler models "
                    + "and attires (.pac), titantron videos, arenas and more. Add the mod's .zip; the original "
                    + "game files are never changed and you can switch mods off at any time. Mods lower in the "
                    + "list win when two change the same file.", 13, DIM));
            gameMods.addView(button("Add mod (.zip)…", v -> pick(REQ_MOD, false)), wrap(dp(8)));
            modList = new LinearLayout(this);
            modList.setOrientation(LinearLayout.VERTICAL);
            gameMods.addView(modList);

        }

        // ---- UPDATES
        currentTab = "UPDATES";
        LinearLayout upd = section(page, "Updates");
        upd.addView(checkbox("Check for updates when the launcher opens", prefs.getBoolean("check_updates", true),
                c -> prefs.edit().putBoolean("check_updates", c).apply()));
        upd.addView(button("Check for updates now", v -> Updater.check(this, false, this::showUpdate)), wrap(dp(4)));

        // ---- SAVES
        currentTab = "SAVES";
        LinearLayout saves = section(page, "Saves");
        saves.addView(checkbox("Back up my saves before each start (the newest 20 are kept)", prefs.getBoolean("auto_backup", true),
                c -> prefs.edit().putBoolean("auto_backup", c).apply()));
        saves.addView(button("Back up my saves now", v -> {
            String err = LauncherNative.backupSaves();
            toastDialog(err.isEmpty() ? "Saves backed up." : err);
            refresh();
        }), wrap(dp(4)));
        backupList = new LinearLayout(this);
        backupList.setOrientation(LinearLayout.VERTICAL);
        saves.addView(backupList);

        if (Features.EXTRAS) {
            // Save sync
            LinearLayout sync = section(page, "Save sync between devices");
            sync.addView(text("Keep one sync file in Google Drive (or any cloud folder). The launcher uploads your "
                    + "saves when you leave the game and downloads newer saves before you play. The file is a "
                    + "normal save backup, so the PC launcher can restore it too.", 13, DIM));
            syncStatus = text("", 14, TEXT);
            sync.addView(syncStatus);
            LinearLayout syncButtons = row();
            syncButtons.addView(button("Create sync file…", v -> {
                Intent i = new Intent(Intent.ACTION_CREATE_DOCUMENT);
                i.addCategory(Intent.CATEGORY_OPENABLE);
                i.setType("application/zip");
                i.putExtra(Intent.EXTRA_TITLE, "WWE13-saves-sync.zip");
                startActivityForResult(i, REQ_SYNC_NEW);
            }));
            syncButtons.addView(button("Use existing sync file…", v -> {
                Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT);
                i.addCategory(Intent.CATEGORY_OPENABLE);
                i.setType("*/*");
                startActivityForResult(i, REQ_SYNC_OPEN);
            }));
            syncButtons.addView(button("Sync now", v -> {
                if (!SaveSync.isSetUp(this)) toastDialog("Create a sync file or choose an existing one first.");
                else runSync(true);
            }));
            sync.addView(syncButtons);
            sync.addView(checkbox("Sync automatically", prefs.getBoolean("sync_auto", true),
                    c -> prefs.edit().putBoolean("sync_auto", c).apply()));

        }

        // The menu on the left: one entry per page that has something on it.
        for (String name : new String[] {"PLAY", "GRAPHICS", "GAME", "CONTROLS", "MUSIC", "MODS", "SAVES", "UPDATES"}) {
            if (!tabs.containsKey(name)) continue;
            final String n = name;
            Button b = navButton(name, v -> showTab(n));
            tabButtons.put(name, b);
            navItems.addView(b, matchWidth(dp(2)));
        }
        setContentView(rootView);
        refresh();
        showTab(prefs.getString("launcher_tab", "PLAY"));
        playButton.requestFocus();
    }

    // ------------------------------------------------------------------------------------------ pages
    private final java.util.LinkedHashMap<String, List<View>> tabs = new java.util.LinkedHashMap<>();
    private final java.util.HashMap<String, Button> tabButtons = new java.util.HashMap<>();
    private String currentTab = "PLAY", shownTab = "PLAY";
    private LinearLayout navItems;
    private ScrollView contentScroll;

    private void tag(View v) {
        List<View> l = tabs.get(currentTab);
        if (l == null) tabs.put(currentTab, l = new ArrayList<>());
        l.add(v);
    }

    private void showTab(String name) {
        if (!tabs.containsKey(name)) name = "PLAY";
        shownTab = name;
        prefs.edit().putString("launcher_tab", name).apply();
        for (java.util.Map.Entry<String, List<View>> e : tabs.entrySet())
            for (View v : e.getValue()) v.setVisibility(e.getKey().equals(name) ? View.VISIBLE : View.GONE);
        for (java.util.Map.Entry<String, Button> e : tabButtons.entrySet()) e.getValue().setSelected(e.getKey().equals(name));
        if (contentScroll != null) contentScroll.scrollTo(0, 0);
    }

    private Button navButton(String label, View.OnClickListener l) {
        Button b = new Button(this);
        b.setText(label);
        b.setAllCaps(true);
        b.setGravity(Gravity.CENTER_VERTICAL | Gravity.START);
        b.setTypeface(Typeface.create("sans-serif-condensed", Typeface.BOLD_ITALIC));
        b.setTextSize(TypedValue.COMPLEX_UNIT_SP, 21);
        b.setLetterSpacing(0.04f);
        b.setPadding(dp(14), dp(6), dp(10), dp(6));
        b.setMinHeight(0);
        b.setMinimumHeight(dp(44));
        int[][] st = {{android.R.attr.state_focused}, {android.R.attr.state_pressed}, {android.R.attr.state_selected}, {}};
        b.setTextColor(new android.content.res.ColorStateList(st, new int[] {Color.WHITE, Color.WHITE, ACCENT, 0xFFBDBDBD}));
        StateListDrawable d = new StateListDrawable();
        d.addState(new int[] {android.R.attr.state_focused}, slant(ACCENT));
        d.addState(new int[] {android.R.attr.state_pressed}, slant(ACCENT));
        d.addState(new int[] {}, new android.graphics.drawable.ColorDrawable(Color.TRANSPARENT));
        b.setBackground(d);
        b.setOnClickListener(l);
        b.setOnFocusChangeListener((v, f) -> { if (f) showTab(label); });
        b.setFocusable(true);
        return b;
    }

    /** A red bar with a slanted right end, like the highlight in the game's menus. */
    private android.graphics.drawable.Drawable slant(int color) {
        return new android.graphics.drawable.Drawable() {
            final android.graphics.Paint p = new android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG);
            @Override public void draw(android.graphics.Canvas c) {
                android.graphics.Rect r = getBounds();
                p.setColor(color);
                android.graphics.Path path = new android.graphics.Path();
                float k = r.height() * 0.35f;
                path.moveTo(r.left, r.top);
                path.lineTo(r.right, r.top);
                path.lineTo(r.right - k, r.bottom);
                path.lineTo(r.left, r.bottom);
                path.close();
                c.drawPath(path, p);
            }
            @Override public void setAlpha(int a) {}
            @Override public void setColorFilter(android.graphics.ColorFilter f) {}
            @Override public int getOpacity() { return android.graphics.PixelFormat.TRANSLUCENT; }
        };
    }

    /** Black with the game's big red X and a dark vignette. */
    private final class XBackground extends android.graphics.drawable.Drawable {
        final android.graphics.Paint p = new android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG);
        @Override public void draw(android.graphics.Canvas c) {
            android.graphics.Rect r = getBounds();
            c.drawColor(BG);
            float w = r.width(), h = r.height(), cx = w * 0.62f, cy = h * 0.5f, t = h * 0.16f;
            p.setStyle(android.graphics.Paint.Style.FILL);
            p.setShader(new android.graphics.LinearGradient(0, 0, w, h, 0x22C0141A, 0x99C0141A, android.graphics.Shader.TileMode.CLAMP));
            for (int k = 0; k < 2; ++k) {
                android.graphics.Path band = new android.graphics.Path();
                float dx = h * 0.75f * (k == 0 ? 1 : -1);
                band.moveTo(cx - dx - t, cy - h);
                band.lineTo(cx - dx + t, cy - h);
                band.lineTo(cx + dx + t, cy + h);
                band.lineTo(cx + dx - t, cy + h);
                band.close();
                c.drawPath(band, p);
            }
            p.setShader(new android.graphics.RadialGradient(cx, cy, Math.max(w, h) * 0.75f, 0x00000000, 0xE6000000,
                    android.graphics.Shader.TileMode.CLAMP));
            c.drawRect(r, p);
            p.setShader(null);
        }
        @Override public void setAlpha(int a) {}
        @Override public void setColorFilter(android.graphics.ColorFilter f) {}
        @Override public int getOpacity() { return android.graphics.PixelFormat.OPAQUE; }
    }

    private void refresh() {
        String[] s = LauncherNative.status().split("\n", 2);
        String[] f = s[0].split("\\|");
        boolean base = f[0].equals("found");
        boolean tu = f.length > 1 && f[1].equals("found");
        StringBuilder sb = new StringBuilder();
        sb.append(base ? "✔ Game installed" : "✘ Game not installed — install it from your disc image");
        sb.append("\n");
        sb.append(tu ? "✔ Title update 2.0.1.0" : "✘ Title update 2.0.1.0 missing — add the title update file");
        if (f.length > 3) sb.append("\nDLC packs: ").append(f[2]).append(" of ").append(f[3]);
        statusText.setText(sb.toString());
        playButton.setEnabled(base && tu);
        playButton.setAlpha(base && tu ? 1f : 0.4f);

        if (songList != null) songList.removeAllViews();
        if (songList != null) for (String line : lines(LauncherNative.listSongs())) {
            final String[] p = line.split("\\|");
            if (p.length < 2) continue;
            songList.addView(listItem("♪  " + p[1] + "  (" + (p.length > 2 ? p[2] : "") + ")", "Remove", () -> {
                LauncherNative.removeSong(p[0]);
                refresh();
            }));
        }
        if (creationList != null) creationList.removeAllViews();
        if (creationList != null) for (String line : lines(LauncherNative.listCreations())) {
            final String[] p = line.split("\\|");
            if (p.length < 3) continue;
            creationList.addView(listItem(p[0] + ": " + p[1], "Remove", () -> {
                String err = LauncherNative.removeCreation(p[2]);
                if (!err.isEmpty()) toastDialog(err);
                refresh();
            }));
        }
        refreshMods();
        if (syncStatus != null) {
            String name = SaveSync.syncName(this);
            syncStatus.setText(name.isEmpty() ? "Not set up." : "Sync file: " + name
                    + "\nLast result: " + prefs.getString("sync_last", "not synced yet"));
        }
        backupList.removeAllViews();
        List<String> all = new ArrayList<>();
        for (String line : lines(LauncherNative.listBackups())) if (line.split("\\|").length >= 2) all.add(line);
        // Newest first; only the last three unless "Show all" was pressed.
        all.sort((x, y) -> y.split("\\|")[1].compareTo(x.split("\\|")[1]));
        int shown = showAllBackups ? all.size() : Math.min(3, all.size());
        for (String line : all.subList(0, shown)) {
            final String[] p = line.split("\\|");
            String label = p[1].replace('T', ' ') + (p.length > 3 && p[3].equals("1") ? "  (automatic)" : "");
            backupList.addView(listItem(label, "Restore", () -> confirm("Restore this backup? Your current saves are backed up first.", () -> {
                String err = LauncherNative.restoreBackup(p[0]);
                toastDialog(err.isEmpty() ? "Backup restored." : err);
                refresh();
            })));
        }
        LinearLayout more = row();
        if (all.size() > 3) {
            more.addView(button(showAllBackups ? "Show fewer" : "Show all (" + all.size() + ")", v -> {
                showAllBackups = !showAllBackups;
                refresh();
            }));
        }
        if (all.size() > 20) {
            more.addView(button("Delete old backups (keep the newest 20)", v -> confirm(
                    "Delete " + (all.size() - 20) + " old save backups? The newest 20 are kept. This can't be undone.", () -> {
                        int n = LauncherNative.deleteOldBackups(20);
                        toastDialog(n + " old backups deleted.");
                        refresh();
                    })));
        }
        if (more.getChildCount() > 0) backupList.addView(more, wrap(dp(4)));
    }
    private boolean showAllBackups;

    private boolean gameReady() {
        String[] f = LauncherNative.status().split("\n", 2)[0].split("\\|");
        return f.length > 1 && f[0].equals("found") && f[1].equals("found");
    }

    // ------------------------------------------------------------------------------------------ launch

    private void startGame() {
        if (Features.EXTRAS && SaveSync.isSetUp(this) && prefs.getBoolean("sync_auto", true) && !syncedBeforePlay) {
            syncedBeforePlay = true;
            runSync(false, this::startGame);
            return;
        }
        try {
            writeEnabledMods();
            String[] resArg = {"--resolution=720p", "--resolution=720p", "--resolution=720p", "--resolution=1080p", "--resolution=720p"};
            String[] internal = {"480", "576", null, "1080", null};
            int res = prefs.getInt("resolution", 2);
            StringBuilder opts = new StringBuilder();
            opts.append(resArg[res]).append('\n');
            if (res == 4) opts.append("--resolution_scale=2\n");
            if (prefs.getBoolean("stretch", false)) opts.append("--present_letterbox=false\n");
            write(new File(root, "launch_options.txt"), opts.toString());

            StringBuilder env = new StringBuilder();
            env.append("WWE13_INTERNAL_RES=").append(internal[res] != null ? internal[res] : "").append('\n');
            env.append("WWE13_SCENE_AA=").append(prefs.getBoolean("aa_faster", false) ? "2x" : "").append('\n');
            int fps = prefs.getInt("frame_rate", 1);
            env.append("WWE13_KEEP_60=").append(fps == 1 ? "1" : "").append('\n');
            env.append("WWE13_LOCK_30=").append(fps == 2 ? "1" : "").append('\n');
            env.append("WWE13_UNLOCK_ALL=").append(prefs.getBoolean("unlock_all", false) ? "1" : "").append('\n');
            env.append("WWE13_TRAINING=").append(Features.EXTRAS && prefs.getBoolean("training", false) ? "1" : "").append('\n');
            env.append("WWE13_PS_PROMPTS=").append(prefs.getInt("button_icons", 0) == 1 ? "1" : "").append('\n');
            env.append("WWE13_AI_KICKOUT=").append(new String[] {"", "55", "10"}[Math.max(0, Math.min(2, prefs.getInt("cpu_kickouts", 0)))]).append('\n');
            env.append("WWE13_SHOW_FPS=").append(prefs.getBoolean("show_fps", false) ? "1" : "").append('\n');
            write(new File(root, "env.txt"), env.toString());

            File drivers = new File(getFilesDir(), "drivers");
            drivers.mkdirs();
            write(new File(drivers, "current.txt"), currentDriver());

            if (prefs.getBoolean("auto_backup", true)) LauncherNative.backupSavesAuto();
        } catch (Exception e) {
            toastDialog("Could not save the settings: " + e.getMessage());
            return;
        }
        startActivity(new Intent(this, WWE13Activity.class));
        finish();
    }

    private boolean syncedBeforePlay = false;

    private void recordSync(SaveSync.Result r) {
        if (r.outcome == SaveSync.Outcome.NOT_SET_UP) return;
        String when = java.text.DateFormat.getDateTimeInstance(java.text.DateFormat.SHORT, java.text.DateFormat.SHORT)
                .format(new java.util.Date());
        prefs.edit().putString("sync_last", when + " – " + r.message).apply();
    }

    private void runSync(boolean interactive) {
        runSync(interactive, null);
    }

    /** Syncs on a background thread; asks about conflicts; then runs next (if any). */
    private void runSync(boolean interactive, Runnable next) {
        AlertDialog wait = new AlertDialog.Builder(this, android.R.style.Theme_DeviceDefault_Dialog_Alert)
                .setMessage("Syncing saves…").setCancelable(false).create();
        wait.show();
        new Thread(() -> {
            SaveSync.Result r = SaveSync.sync(this, null, false);
            handler.post(() -> {
                wait.dismiss();
                recordSync(r);
                if (r.outcome == SaveSync.Outcome.CONFLICT) {
                    new AlertDialog.Builder(this, android.R.style.Theme_DeviceDefault_Dialog_Alert)
                            .setTitle("Which saves do you want to keep?")
                            .setMessage(r.message + " The other ones are kept as a backup.")
                            .setPositiveButton("This device", (d, w) -> resolveSync(true, next))
                            .setNegativeButton("Sync file", (d, w) -> resolveSync(false, next))
                            .setNeutralButton("Decide later", (d, w) -> { if (next != null) next.run(); })
                            .setCancelable(false).show();
                } else {
                    if (interactive) toastDialogThen(r.message, next);
                    else if (r.outcome == SaveSync.Outcome.FAILED && !r.message.equals(prefs.getString("sync_fail_shown", ""))) {
                        // (an automatic sync that fails is shown once, not at every start)
                        prefs.edit().putString("sync_fail_shown", r.message).apply();
                        boolean gone = r.message.contains("Permission Denial") || r.message.contains("FileNotFound")
                                || r.message.contains("No such file");
                        toastDialogThen(gone ? "The sync file can't be reached any more (moved or deleted?). Pick it again "
                                + "under \"Save sync between devices\" with \"Use existing sync file…\", or make a new one."
                                : r.message, next);
                    } else if (next != null) next.run();
                    if (r.outcome != SaveSync.Outcome.FAILED) prefs.edit().remove("sync_fail_shown").apply();
                    if (syncStatus != null) refresh();
                }
            });
        }).start();
    }

    private void resolveSync(boolean keepDevice, Runnable next) {
        if (!keepDevice) LauncherNative.backupSavesAuto();
        new Thread(() -> {
            SaveSync.Result r = SaveSync.sync(this, keepDevice, false);
            handler.post(() -> {
                recordSync(r);
                toastDialogThen(r.message, next);
                if (syncStatus != null) refresh();
            });
        }).start();
    }

    private void toastDialogThen(String message, Runnable next) {
        new AlertDialog.Builder(this, android.R.style.Theme_DeviceDefault_Dialog_Alert)
                .setMessage(message).setCancelable(false)
                .setPositiveButton("OK", (d, w) -> { if (next != null) next.run(); }).show();
    }

    // ------------------------------------------------------------------------------------------ GPU drivers

    /** Bundled Turnip first, then the phone's own driver, then drivers the player added. */
    private List<String> driverIds() {
        List<String> ids = new ArrayList<>();
        ids.add(BUNDLED_DRIVER);
        ids.add("system");
        File[] dirs = new File(getFilesDir(), "drivers").listFiles(File::isDirectory);
        if (dirs != null) {
            List<String> extra = new ArrayList<>();
            for (File d : dirs) if (!d.getName().equals(BUNDLED_DRIVER)) extra.add(d.getName());
            java.util.Collections.sort(extra, String.CASE_INSENSITIVE_ORDER);
            ids.addAll(extra);
        }
        return ids;
    }

    /** Adreno 6xx (Snapdragon 865 / 870 / 888 ...): Turnip renders there now, but hangs the GPU after a minute or two
     *  (Retroid Pocket Flip 2), so these default to the phone's own driver. */
    private static Boolean adreno6xx;
    static boolean isAdreno6xx() {
        if (adreno6xx == null) {
            String model = "";
            try (java.io.BufferedReader r = new java.io.BufferedReader(new java.io.FileReader("/sys/class/kgsl/kgsl-3d0/gpu_model"))) {
                String line = r.readLine();
                if (line != null) model = line.trim();
            } catch (Exception ignored) {
            }
            boolean six = model.startsWith("Adreno6");
            if (model.isEmpty()) {
                String board = android.os.Build.BOARD == null ? "" : android.os.Build.BOARD.toLowerCase(java.util.Locale.ROOT);
                String hw = android.os.Build.HARDWARE == null ? "" : android.os.Build.HARDWARE.toLowerCase(java.util.Locale.ROOT);
                for (String p : new String[] {"kona", "lahaina", "msmnile", "lito"}) if (board.equals(p) || hw.equals(p)) six = true;
            }
            adreno6xx = six;
        }
        return adreno6xx;
    }

    private String driverLabel(String id) {
        if (id.equals(BUNDLED_DRIVER)) return isAdreno6xx() ? "Turnip T30 (faster, may crash on this GPU)" : "Turnip T30 (recommended)";
        if (id.equals("system")) return isAdreno6xx() ? "Phone's own driver (recommended)" : "Phone's own driver";
        return id.replace('_', ' ');
    }

    private String currentDriver() {
        String id = prefs.getString("driver_name", null);
        if (id == null && prefs.contains("driver")) id = prefs.getInt("driver", 0) == 1 ? "system" : BUNDLED_DRIVER;  // older settings
        if (id == null) id = isAdreno6xx() ? "system" : BUNDLED_DRIVER;
        if (!id.equals("system") && !new File(getFilesDir(), "drivers/" + id).isDirectory()) id = BUNDLED_DRIVER;
        return id;
    }

    /** Installs an adrenotools driver package (.zip with meta.json and the driver library). */
    private void installDriver(Uri uri) {
        runLong("Installing the GPU driver", () -> {
            String file = displayName(uri);
            String base = file.replaceAll("(?i)\\.zip$", "").replaceAll("[^A-Za-z0-9._-]", "_");
            if (base.isEmpty() || base.equals("system")) base = "driver";
            File dest = new File(getFilesDir(), "drivers/" + base);
            deleteTree(dest);
            dest.mkdirs();
            boolean hasLibrary = false;
            try (ZipInputStream zin = new ZipInputStream(getContentResolver().openInputStream(uri))) {
                ZipEntry e;
                while ((e = zin.getNextEntry()) != null) {
                    if (e.isDirectory()) continue;
                    String name = e.getName().replace('\\', '/');
                    name = name.substring(name.lastIndexOf('/') + 1);
                    if (name.isEmpty()) continue;
                    if (name.endsWith(".so")) hasLibrary = true;
                    try (OutputStream os = new FileOutputStream(new File(dest, name))) {
                        copy(zin, os);
                    }
                }
            }
            if (!hasLibrary) {
                deleteTree(dest);
                return "This zip doesn't contain a GPU driver (no .so library found).";
            }
            prefs.edit().putString("driver_name", base).apply();
            handler.post(this::buildUi);
            return "Driver installed and selected: " + base.replace('_', ' ')
                    + "\nIf the game doesn't start or looks wrong, switch back to Turnip T30.";
        });
    }

    // ------------------------------------------------------------------------------------------ WWE 2K14 superstars

    /** Runs the porter (libk14port.so, a program in the app's native library folder) on the player's 2K14 disc. */
    private void portK14(Uri uri) {
        if (!new File(root, "WWE 13/pac").isDirectory()) {
            toastDialog("Install WWE '13 first (\"Install from my disc image\"), then port the WWE 2K14 superstars.");
            return;
        }
        LinearLayout box = new LinearLayout(this);
        box.setOrientation(LinearLayout.VERTICAL);
        box.setPadding(dp(20), dp(16), dp(20), dp(8));
        TextView label = text("Starting…", 14, TEXT);
        ProgressBar bar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        bar.setMax(100);
        box.addView(label);
        box.addView(bar);
        AlertDialog dlg = new AlertDialog.Builder(this, android.R.style.Theme_DeviceDefault_Dialog_Alert)
                .setTitle("Porting WWE 2K14 superstars").setView(box).setCancelable(false).create();
        dlg.show();
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        new Thread(() -> {
            String result;
            File out = new File(modsDir(), K14_MOD + ".part");
            deleteTree(out);
            try (ParcelFileDescriptor pfd = getContentResolver().openFileDescriptor(uri, "r")) {
                String iso = "/proc/" + android.os.Process.myPid() + "/fd/" + pfd.getFd();
                File exe = new File(getApplicationInfo().nativeLibraryDir, "libk14port.so");
                Process p = new ProcessBuilder(exe.getAbsolutePath(), "--w13", new File(root, "WWE 13").getAbsolutePath(),
                        "--iso", iso, "--out", out.getAbsolutePath()).redirectErrorStream(true).start();
                java.io.BufferedReader r = new java.io.BufferedReader(new java.io.InputStreamReader(p.getInputStream()));
                String line, last = "";
                int done = -1;
                while ((line = r.readLine()) != null) {
                    if (line.startsWith("progress ")) {
                        String[] f = line.split(" ", 3);
                        final int pct = Integer.parseInt(f[1]);
                        final String what = f.length > 2 ? f[2] : "";
                        handler.post(() -> { bar.setProgress(pct); label.setText(what); });
                    } else if (line.startsWith("done ")) {
                        done = Integer.parseInt(line.substring(5).trim());
                    } else if (line.startsWith("error ")) {
                        last = line.substring(6);
                    }
                }
                p.waitFor();
                if (done > 0) {
                    File dst = new File(modsDir(), K14_MOD);
                    deleteTree(dst);
                    if (!out.renameTo(dst)) throw new Exception("couldn't finish the mod folder");
                    prefs.edit().remove("mod_off_" + K14_MOD).apply();
                    writeEnabledMods();
                    result = done + " WWE 2K14 superstars added. In the game, press R3 on the superstar select screen.";
                } else {
                    deleteTree(out);
                    result = "Porting stopped: " + (last.isEmpty() ? "unknown error" : last);
                }
            } catch (Exception e) {
                deleteTree(out);
                result = "Porting stopped: " + e.getMessage();
            }
            final String msg = result;
            handler.post(() -> {
                getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
                dlg.dismiss();
                toastDialog(msg);
                refreshMods();
            });
        }).start();
    }

    // ------------------------------------------------------------------------------------------ game file mods

    private File modsDir() {
        File d = new File(root, "mods");
        d.mkdirs();
        return d;
    }

    /** Mod folders in load order: enabled.txt order first, then any others alphabetically. */
    private List<String> modOrder() {
        List<String> order = new ArrayList<>();
        File[] dirs = modsDir().listFiles(File::isDirectory);
        List<String> present = new ArrayList<>();
        if (dirs != null) for (File d : dirs) present.add(d.getName());
        java.util.Collections.sort(present, String.CASE_INSENSITIVE_ORDER);
        for (String line : prefs.getString("mod_order", "").split("\n")) {
            if (present.contains(line) && !order.contains(line)) order.add(line);
        }
        for (String n : present) if (!order.contains(n)) order.add(n);
        return order;
    }

    private boolean modEnabled(String name) {
        return !prefs.getBoolean("mod_off_" + name, false);
    }

    private void saveModOrder(List<String> order) {
        prefs.edit().putString("mod_order", android.text.TextUtils.join("\n", order)).apply();
        try {
            writeEnabledMods();
        } catch (Exception ignored) {
        }
    }

    private void writeEnabledMods() throws Exception {
        StringBuilder sb = new StringBuilder();
        // (release build: no mods are loaded; the mod folders are kept as they are)
        if (Features.EXTRAS) for (String n : modOrder()) if (modEnabled(n)) sb.append(n).append('\n');
        write(new File(modsDir(), "enabled.txt"), sb.toString());
    }

    private void refreshMods() {
        if (modList == null) return;
        modList.removeAllViews();
        final List<String> order = modOrder();
        for (int i = 0; i < order.size(); ++i) {
            final String name = order.get(i);
            final int index = i;
            LinearLayout r = row();
            r.setGravity(Gravity.CENTER_VERTICAL);
            CheckBox c = checkbox(name + "  (" + countFiles(new File(modsDir(), name)) + " files)", modEnabled(name), on -> {
                prefs.edit().putBoolean("mod_off_" + name, !on).apply();
                saveModOrder(order);
            });
            r.addView(c, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
            if (index > 0) {
                Button up = button("▲", v -> {
                    order.remove(index);
                    order.add(index - 1, name);
                    saveModOrder(order);
                    refreshMods();
                });
                r.addView(up);
            }
            Button remove = button("Remove", v -> confirm("Remove the mod \"" + name + "\"?", () -> {
                deleteTree(new File(modsDir(), name));
                prefs.edit().remove("mod_off_" + name).apply();
                order.remove(name);
                saveModOrder(order);
                refreshMods();
            }));
            remove.setTextSize(TypedValue.COMPLEX_UNIT_SP, 13);
            r.addView(remove);
            modList.addView(r);
        }
    }

    private static int countFiles(File dir) {
        File[] list = dir.listFiles();
        if (list == null) return 0;
        int n = 0;
        for (File f : list) n += f.isDirectory() ? countFiles(f) : 1;
        return n;
    }

    private static void deleteTree(File f) {
        File[] list = f.listFiles();
        if (list != null) for (File c : list) deleteTree(c);
        f.delete();
    }

    /** Lower-case names of the game's top-level files and folders, to find where a mod's files start. */
    private java.util.Set<String> gameTopLevel() {
        java.util.Set<String> names = new java.util.HashSet<>();
        String[] list = new File(root, "WWE 13").list();
        if (list != null) for (String n : list) names.add(n.toLowerCase(Locale.ROOT));
        return names;
    }

    private void installMod(Uri uri) {
        runLong("Installing the mod", () -> {
            String file = displayName(uri);
            if (!file.toLowerCase(Locale.ROOT).endsWith(".zip")) {
                return "Please pick the mod as a .zip file. (RAR and 7z mods need to be repacked as .zip first.)";
            }
            String base = file.substring(0, file.length() - 4).replaceAll("[\\\\/:*?\"<>|]", "_").trim();
            if (base.isEmpty()) base = "mod";
            File dest = new File(modsDir(), base);
            for (int i = 2; dest.exists(); ++i) dest = new File(modsDir(), base + " " + i);
            java.util.Set<String> top = gameTopLevel();
            int installed = 0;
            List<String> skipped = new ArrayList<>();
            try (ZipInputStream zin = new ZipInputStream(getContentResolver().openInputStream(uri))) {
                ZipEntry e;
                while ((e = zin.getNextEntry()) != null) {
                    if (e.isDirectory()) continue;
                    String[] parts = e.getName().replace('\\', '/').split("/");
                    int start = -1;
                    for (int i = 0; i < parts.length; ++i) {
                        if (top.contains(parts[i].toLowerCase(Locale.ROOT))) { start = i; break; }
                    }
                    String rel;
                    String leaf = parts[parts.length - 1];
                    String lower = leaf.toLowerCase(Locale.ROOT);
                    if (start >= 0) {
                        StringBuilder sb = new StringBuilder();
                        for (int i = start; i < parts.length; ++i) {
                            if (parts[i].equals("..") || parts[i].isEmpty()) { sb.setLength(0); break; }
                            if (sb.length() > 0) sb.append('/');
                            sb.append(parts[i]);
                        }
                        rel = sb.toString();
                    } else if (lower.endsWith(".pac")) {
                        rel = "pac/" + leaf;  // loose wrestler/attire files belong in the pac folder
                    } else {
                        rel = "";
                    }
                    if (rel.isEmpty()) {
                        if (!lower.endsWith(".txt") && !lower.endsWith(".png") && !lower.endsWith(".jpg")) skipped.add(leaf);
                        continue;
                    }
                    File out = new File(dest, rel);
                    out.getParentFile().mkdirs();
                    try (OutputStream os = new FileOutputStream(out)) {
                        copy(zin, os);
                    }
                    ++installed;
                }
            }
            if (installed == 0) {
                deleteTree(dest);
                return "No game files were found in this mod." + (skipped.isEmpty() ? "" : "\nIt contains: "
                        + android.text.TextUtils.join(", ", skipped.subList(0, Math.min(8, skipped.size()))));
            }
            List<String> order = modOrder();
            saveModOrder(order);
            return "Installed \"" + dest.getName() + "\" (" + installed + " files)."
                    + (skipped.isEmpty() ? "" : "\nSkipped files the game doesn't use: "
                    + android.text.TextUtils.join(", ", skipped.subList(0, Math.min(8, skipped.size()))));
        });
    }

    private static void write(File f, String s) throws Exception {
        try (OutputStream out = new FileOutputStream(f)) {
            out.write(s.getBytes("UTF-8"));
        }
    }

    // ------------------------------------------------------------------------------------------ file picking

    private void pick(int request, boolean multiple) {
        Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        i.addCategory(Intent.CATEGORY_OPENABLE);
        i.setType(request == REQ_SONGS ? "audio/*" : "*/*");
        if (multiple) i.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true);
        startActivityForResult(i, request);
    }

    @Override
    protected void onActivityResult(int request, int result, Intent data) {
        super.onActivityResult(request, result, data);
        if (result != RESULT_OK || data == null) return;
        List<Uri> uris = new ArrayList<>();
        ClipData clip = data.getClipData();
        if (clip != null) {
            for (int i = 0; i < clip.getItemCount(); ++i) uris.add(clip.getItemAt(i).getUri());
        } else if (data.getData() != null) {
            uris.add(data.getData());
        }
        if (uris.isEmpty()) return;
        if (request == REQ_SYNC_NEW || request == REQ_SYNC_OPEN) {
            Uri u = uris.get(0);
            try {
                getContentResolver().takePersistableUriPermission(u,
                        Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
            } catch (Exception e) {
                toastDialog("This location can't be kept for syncing. Try a file in Google Drive or on this device.");
                return;
            }
            prefs.edit().putString("sync_uri", u.toString()).remove("sync_local_fp").remove("sync_remote_sha").apply();
            runSync(true);
            return;
        }
        switch (request) {
            case REQ_ISO: installIso(uris.get(0)); break;
            case REQ_PACKAGE: installPackages(uris); break;
            case REQ_SONGS: addSongs(uris); break;
            case REQ_CREATIONS: importCreations(uris); break;
            case REQ_MOD: installMod(uris.get(0)); break;
            case REQ_DRIVER: installDriver(uris.get(0)); break;
            case REQ_K14: portK14(uris.get(0)); break;
        }
    }

    private void installIso(Uri uri) {
        runLong("Installing the game from your disc image", () -> {
            try (ParcelFileDescriptor pfd = getContentResolver().openFileDescriptor(uri, "r")) {
                return LauncherNative.extractIso("/proc/self/fd/" + pfd.getFd());
            }
        });
    }

    private void installPackages(List<Uri> uris) {
        runLong("Installing the title update / DLC", () -> {
            StringBuilder messages = new StringBuilder();
            for (Uri uri : uris) {
                File copy = copyToCache(uri, "packages");
                String[] info = LauncherNative.inspectPackage(copy.getAbsolutePath()).split("\\|", 2);
                if (info[0].equals("unknown")) {
                    messages.append(copy.getName()).append(": not a WWE '13 title update or DLC pack.\n");
                } else {
                    String err = LauncherNative.importPackage(copy.getAbsolutePath());
                    messages.append(info.length > 1 && !info[1].isEmpty() ? info[1] : copy.getName()).append(": ")
                            .append(err.isEmpty() ? "installed" : err).append('\n');
                }
                copy.delete();
            }
            return messages.toString().trim();
        });
    }

    private void addSongs(List<Uri> uris) {
        runLong("Adding songs", () -> {
            File musicDir = new File(root, "music");
            musicDir.mkdirs();
            int added = 0;
            StringBuilder skipped = new StringBuilder();
            for (Uri uri : uris) {
                String name = displayName(uri);
                String lower = name.toLowerCase(Locale.ROOT);
                if (!(lower.endsWith(".mp3") || lower.endsWith(".flac") || lower.endsWith(".ogg") || lower.endsWith(".wav"))) {
                    skipped.append(name).append(" (not MP3, FLAC, OGG or WAV)\n");
                    continue;
                }
                File dest = new File(musicDir, name);
                try (InputStream in = getContentResolver().openInputStream(uri); OutputStream out = new FileOutputStream(dest)) {
                    copy(in, out);
                    ++added;
                }
            }
            return added + " song(s) added." + (skipped.length() > 0 ? "\nSkipped:\n" + skipped : "");
        });
    }

    private void importCreations(List<Uri> uris) {
        runLong("Importing creations", () -> {
            List<String> paths = new ArrayList<>();
            List<File> copies = new ArrayList<>();
            for (Uri uri : uris) {
                File copy = copyToCache(uri, "creations");
                copies.add(copy);
                paths.add(copy.getAbsolutePath());
            }
            String pack = copies.isEmpty() ? "" : copies.get(0).getName();
            String err = LauncherNative.importCreations(paths.toArray(new String[0]), pack, false);
            for (File c : copies) c.delete();
            return err.isEmpty() ? copies.size() + " creation(s) installed." : err;
        });
    }

    private File copyToCache(Uri uri, String sub) throws Exception {
        File dir = new File(getCacheDir(), sub);
        dir.mkdirs();
        File dest = new File(dir, displayName(uri));
        try (InputStream in = getContentResolver().openInputStream(uri); OutputStream out = new FileOutputStream(dest)) {
            copy(in, out);
        }
        return dest;
    }

    private String displayName(Uri uri) {
        try (Cursor c = getContentResolver().query(uri, new String[]{OpenableColumns.DISPLAY_NAME}, null, null, null)) {
            if (c != null && c.moveToFirst()) {
                String n = c.getString(0);
                if (n != null && !n.isEmpty()) return n.replace('/', '_');
            }
        } catch (Exception ignored) {
        }
        return "file_" + System.currentTimeMillis();
    }

    // ------------------------------------------------------------------------------------------ long operations

    interface Job {
        String run() throws Exception;
    }

    private void runLong(String title, Job job) {
        LinearLayout box = new LinearLayout(this);
        box.setOrientation(LinearLayout.VERTICAL);
        box.setPadding(dp(24), dp(16), dp(24), dp(8));
        ProgressBar bar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        bar.setMax(1000);
        TextView label = text("Starting…", 14, TEXT);
        box.addView(bar);
        box.addView(label);
        AlertDialog dialog = new AlertDialog.Builder(this, android.R.style.Theme_DeviceDefault_Dialog_Alert)
                .setTitle(title).setView(box).setCancelable(false)
                .setNegativeButton("Cancel", (d, w) -> LauncherNative.cancel()).create();
        dialog.show();
        Runnable poll = new Runnable() {
            @Override
            public void run() {
                String[] p = LauncherNative.progress().split("\\|", 3);
                try {
                    long done = Long.parseLong(p[0]), total = Long.parseLong(p[1]);
                    if (total > 0) bar.setProgress((int) (done * 1000 / total));
                    bar.setIndeterminate(total <= 0);
                } catch (Exception ignored) {
                }
                if (p.length > 2 && !p[2].isEmpty()) label.setText(p[2]);
                if (dialog.isShowing()) handler.postDelayed(this, 200);
            }
        };
        handler.post(poll);
        new Thread(() -> {
            String message;
            try {
                message = job.run();
            } catch (Exception e) {
                message = "Something went wrong: " + e.getMessage();
            }
            final String m = message;
            handler.post(() -> {
                dialog.dismiss();
                toastDialog(m == null || m.isEmpty() ? "Done." : m);
                refresh();
            });
        }).start();
    }

    // ------------------------------------------------------------------------------------------ small widgets

    interface IntListener {
        void on(int value);
    }

    interface BoolListener {
        void on(boolean value);
    }

    // The newest crash log (crash_logs/crash-*.txt, written by the game when it dies) not yet dismissed: why it
    // ended, and a button to save it to Downloads (to send it on).
    private void addCrashNotice(LinearLayout page) {
        File[] logs = new File(root, "crash_logs").listFiles((d, n) -> n.startsWith("crash-") && n.endsWith(".txt"));
        if (logs == null || logs.length == 0) return;
        File newest = logs[0];
        for (File f : logs) if (f.getName().compareTo(newest.getName()) > 0) newest = f;
        if (newest.getName().equals(prefs.getString("crash_seen", ""))) return;
        String reason = "", fatal = "";
        try (java.io.BufferedReader r = new java.io.BufferedReader(new java.io.FileReader(newest))) {
            String l;
            while ((l = r.readLine()) != null) {
                if (l.startsWith("Reason: ")) reason = l.substring(8);
                int k = l.indexOf("[FATAL]");
                if (k >= 0) fatal = l.substring(k);
            }
        } catch (Exception ignored) {
        }
        final File log = newest;
        LinearLayout box = section(page, "The game crashed last time");
        String when = log.getName().substring(6, log.getName().length() - 4);
        box.addView(text(when + ": " + reason + (fatal.isEmpty() ? "" : "\n" + fatal)
                + "\n\nThe crash log is in Android/data/com.toni.wwe13/files/crash_logs/" + log.getName(), 14, TEXT));
        LinearLayout buttons = row();
        buttons.addView(button("Save crash log to Downloads", v -> {
            try {
                android.content.ContentValues cv = new android.content.ContentValues();
                cv.put(android.provider.MediaStore.Downloads.DISPLAY_NAME, "WWE13-" + log.getName());
                cv.put(android.provider.MediaStore.Downloads.MIME_TYPE, "text/plain");
                android.net.Uri uri = getContentResolver().insert(android.provider.MediaStore.Downloads.EXTERNAL_CONTENT_URI, cv);
                try (java.io.OutputStream o = getContentResolver().openOutputStream(uri);
                     java.io.InputStream i = new java.io.FileInputStream(log)) {
                    byte[] buf = new byte[65536];
                    int n;
                    while ((n = i.read(buf)) > 0) o.write(buf, 0, n);
                }
                toastDialog("Saved to Downloads as WWE13-" + log.getName());
            } catch (Exception e) {
                toastDialog("Couldn't save it: " + e.getMessage());
            }
        }));
        buttons.addView(button("Dismiss", v -> {
            prefs.edit().putString("crash_seen", log.getName()).apply();
            box.setVisibility(View.GONE);
        }));
        box.addView(buttons);
    }

    private LinearLayout section(LinearLayout page, String title) {
        LinearLayout outer = new LinearLayout(this);
        outer.setOrientation(LinearLayout.VERTICAL);
        // The header: a white bar with black italic capitals and a red line under it, as in the game's pause menu.
        TextView t = text(title.toUpperCase(Locale.ROOT), 17, 0xFF0A0A0A);
        t.setTypeface(Typeface.create("sans-serif-condensed", Typeface.BOLD_ITALIC));
        t.setPadding(dp(16), dp(5), dp(16), dp(5));
        t.setBackground(slant(0xFFE8E8E8));
        outer.addView(t, matchWidth(0));
        View red = new View(this);
        red.setBackgroundColor(ACCENT);
        outer.addView(red, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(3)));
        LinearLayout card = new LinearLayout(this);
        card.setOrientation(LinearLayout.VERTICAL);
        card.setPadding(dp(18), dp(10), dp(18), dp(14));
        card.setBackgroundColor(CARD);
        outer.addView(card, matchWidth(0));
        page.addView(outer, matchWidth(dp(14)));
        if (page != updateBox) tag(outer);
        return card;
    }

    private LinearLayout updateBox;

    private void showUpdate(Updater.Release r) {
        if (updateBox == null) return;
        updateBox.removeAllViews();
        LinearLayout box = section(updateBox, "Update available: " + r.tag);
        String notes = r.notes.length() > 600 ? r.notes.substring(0, 600) + "…" : r.notes;
        box.addView(text(notes.replace("**", "").replace("## ", ""), 13, DIM));
        box.addView(button("Download and install " + r.tag + " (" + (r.apkSize >> 20) + " MB)",
                v -> Updater.downloadAndInstall(this, r)), wrap(dp(6)));
        box.addView(text("Your game files, saves and settings stay.", 13, DIM));
    }

    @Override
    protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        Updater.handleStatus(this, intent);
    }

    private TextView text(String s, int sp, int color) {
        TextView t = new TextView(this);
        t.setText(s);
        t.setTextColor(color);
        t.setTextSize(TypedValue.COMPLEX_UNIT_SP, sp);
        t.setPadding(0, dp(4), 0, dp(4));
        return t;
    }

    private Button button(String label, View.OnClickListener l) {
        Button b = new Button(this);
        b.setText(label);
        b.setAllCaps(true);
        b.setTextColor(TEXT);
        b.setTypeface(Typeface.create("sans-serif-condensed", Typeface.BOLD));
        b.setTextSize(TypedValue.COMPLEX_UNIT_SP, 15);
        b.setPadding(dp(18), dp(10), dp(18), dp(10));
        StateListDrawable states = new StateListDrawable();
        states.addState(new int[]{android.R.attr.state_focused}, rounded(ACCENT));
        states.addState(new int[]{android.R.attr.state_pressed}, rounded(ACCENT));
        states.addState(new int[]{}, rounded(0xFF262626));
        b.setBackground(states);
        b.setOnClickListener(l);
        b.setFocusable(true);
        return b;
    }

    private GradientDrawable rounded(int color) {
        GradientDrawable d = new GradientDrawable();
        d.setColor(color);
        d.setCornerRadius(dp(2));
        return d;
    }

    private CheckBox checkbox(String label, boolean checked, BoolListener l) {
        CheckBox c = new CheckBox(this);
        c.setText(label);
        c.setTextColor(TEXT);
        c.setTextSize(TypedValue.COMPLEX_UNIT_SP, 15);
        c.setChecked(checked);
        c.setButtonTintList(android.content.res.ColorStateList.valueOf(ACCENT));
        c.setOnCheckedChangeListener((v, isChecked) -> l.on(isChecked));
        return c;
    }

    private LinearLayout spinnerRow(String label, String[] options, int selected, IntListener l) {
        LinearLayout r = row();
        r.setGravity(Gravity.CENTER_VERTICAL);
        TextView t = text(label, 15, TEXT);
        r.addView(t, new LinearLayout.LayoutParams(dp(180), ViewGroup.LayoutParams.WRAP_CONTENT));
        Spinner s = new Spinner(this, Spinner.MODE_DIALOG);
        ArrayAdapter<String> a = new ArrayAdapter<String>(this, android.R.layout.simple_spinner_item, options) {
            @Override
            public View getView(int position, View convertView, ViewGroup parent) {
                View v = super.getView(position, convertView, parent);
                ((TextView) v).setTextColor(TEXT);
                return v;
            }
        };
        a.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        s.setAdapter(a);
        s.setSelection(Math.max(0, Math.min(selected, options.length - 1)));
        s.setBackground(rounded(0xFF262626));
        s.setOnItemSelectedListener(new android.widget.AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(android.widget.AdapterView<?> p, View v, int pos, long id) {
                l.on(pos);
            }

            @Override
            public void onNothingSelected(android.widget.AdapterView<?> p) {}
        });
        r.addView(s, new LinearLayout.LayoutParams(dp(320), ViewGroup.LayoutParams.WRAP_CONTENT));
        return r;
    }

    private View listItem(String label, String action, Runnable onAction) {
        LinearLayout r = row();
        r.setGravity(Gravity.CENTER_VERTICAL);
        TextView t = text(label, 14, TEXT);
        r.addView(t, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        Button b = button(action, v -> onAction.run());
        b.setTextSize(TypedValue.COMPLEX_UNIT_SP, 13);
        r.addView(b);
        return r;
    }

    private LinearLayout row() {
        LinearLayout r = new LinearLayout(this);
        r.setOrientation(LinearLayout.HORIZONTAL);
        r.setShowDividers(LinearLayout.SHOW_DIVIDER_MIDDLE);
        GradientDrawable gap = new GradientDrawable();
        gap.setSize(dp(10), 1);
        gap.setColor(Color.TRANSPARENT);
        r.setDividerDrawable(gap);
        r.setPadding(0, dp(6), 0, dp(6));
        return r;
    }

    private LinearLayout.LayoutParams matchWidth(int top) {
        LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT);
        p.topMargin = top;
        return p;
    }

    private LinearLayout.LayoutParams wrap(int top) {
        LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT);
        p.topMargin = top;
        return p;
    }

    private void toastDialog(String message) {
        new AlertDialog.Builder(this, android.R.style.Theme_DeviceDefault_Dialog_Alert)
                .setMessage(message).setPositiveButton("OK", null).show();
    }

    private void confirm(String message, Runnable yes) {
        new AlertDialog.Builder(this, android.R.style.Theme_DeviceDefault_Dialog_Alert)
                .setMessage(message).setPositiveButton("Yes", (d, w) -> yes.run()).setNegativeButton("No", null).show();
    }

    private static List<String> lines(String s) {
        List<String> out = new ArrayList<>();
        for (String l : s.split("\n")) if (!l.trim().isEmpty()) out.add(l);
        return out;
    }

    private int dp(int v) {
        return Math.round(v * getResources().getDisplayMetrics().density);
    }
}
