package com.toni.wwe13;

import android.app.Activity;
import android.app.AlertDialog;
import android.app.PendingIntent;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageInstaller;
import android.os.Handler;
import android.os.Looper;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.LinearLayout;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;

/**
 * In-launcher updates: asks GitHub for the latest release of the project, and when it is newer than the installed app
 * downloads its .apk and hands it to Android's installer (the player confirms the install, as Android requires).
 * The game files, saves and settings stay; only the app is replaced.
 */
final class Updater {
    static final String REPO = "dzepdzep/wwe13-recomp-android";

    static final class Release {
        String tag, notes, apkUrl;
        long apkSize;
    }

    interface Found { void onFound(Release r); }

    private Updater() {}

    /** Checks in the background; calls back on the main thread only when a newer release exists. */
    static void check(Activity a, boolean quietIfNone, Found found) {
        new Thread(() -> {
            Release r = null;
            String error = null;
            try {
                r = latest();
            } catch (Exception e) {
                error = e.getMessage();
            }
            final Release fr = r;
            final String fe = error;
            new Handler(Looper.getMainLooper()).post(() -> {
                if (a.isFinishing()) return;
                if (fr != null && isNewer(fr.tag, currentVersion(a))) found.onFound(fr);
                else if (!quietIfNone) {
                    new AlertDialog.Builder(a, android.R.style.Theme_DeviceDefault_Dialog_Alert)
                            .setMessage(fe != null ? "Could not check for updates: " + fe
                                    : "You have the latest version (" + currentVersion(a) + ").")
                            .setPositiveButton("OK", null).show();
                }
            });
        }).start();
    }

    static String currentVersion(Context c) {
        try {
            return c.getPackageManager().getPackageInfo(c.getPackageName(), 0).versionName;
        } catch (Exception e) {
            return "0";
        }
    }

    private static Release latest() throws Exception {
        HttpURLConnection h = (HttpURLConnection) new URL("https://api.github.com/repos/" + REPO + "/releases/latest").openConnection();
        h.setRequestProperty("Accept", "application/vnd.github+json");
        h.setConnectTimeout(10000);
        h.setReadTimeout(15000);
        if (h.getResponseCode() != 200) throw new Exception("GitHub answered " + h.getResponseCode());
        String body;
        try (InputStream in = h.getInputStream()) {
            java.io.ByteArrayOutputStream bo = new java.io.ByteArrayOutputStream();
            byte[] buf = new byte[8192];
            int n;
            while ((n = in.read(buf)) > 0) bo.write(buf, 0, n);
            body = bo.toString("UTF-8");
        }
        JSONObject j = new JSONObject(body);
        Release r = new Release();
        r.tag = j.optString("tag_name", "");
        r.notes = j.optString("body", "");
        JSONArray assets = j.optJSONArray("assets");
        for (int i = 0; assets != null && i < assets.length(); ++i) {
            JSONObject as = assets.getJSONObject(i);
            if (as.optString("name").endsWith(".apk")) {
                r.apkUrl = as.optString("browser_download_url");
                r.apkSize = as.optLong("size");
                break;
            }
        }
        return r.apkUrl == null ? null : r;
    }

    /** "v1.0.10" > "1.0.9": compares the numbers, ignoring a leading v and anything that isn't a digit. */
    static boolean isNewer(String tag, String current) {
        String[] a = tag.replaceAll("[^0-9.]", "").split("\\."), b = current.replaceAll("[^0-9.]", "").split("\\.");
        for (int i = 0; i < Math.max(a.length, b.length); ++i) {
            int x = i < a.length && !a[i].isEmpty() ? Integer.parseInt(a[i]) : 0;
            int y = i < b.length && !b[i].isEmpty() ? Integer.parseInt(b[i]) : 0;
            if (x != y) return x > y;
        }
        return false;
    }

    /** Downloads the release's .apk with a progress dialog, then starts the install. */
    static void downloadAndInstall(Activity a, Release r) {
        LinearLayout box = new LinearLayout(a);
        box.setOrientation(LinearLayout.VERTICAL);
        int pad = (int) (20 * a.getResources().getDisplayMetrics().density);
        box.setPadding(pad, pad / 2, pad, 0);
        ProgressBar bar = new ProgressBar(a, null, android.R.attr.progressBarStyleHorizontal);
        bar.setMax(1000);
        TextView label = new TextView(a);
        box.addView(bar);
        box.addView(label);
        final boolean[] cancelled = {false};
        AlertDialog dialog = new AlertDialog.Builder(a, android.R.style.Theme_DeviceDefault_Dialog_Alert)
                .setTitle("Downloading " + r.tag).setView(box).setCancelable(false)
                .setNegativeButton("Cancel", (d, w) -> cancelled[0] = true).create();
        dialog.show();
        Handler ui = new Handler(Looper.getMainLooper());
        new Thread(() -> {
            File apk = new File(a.getCacheDir(), "update.apk");
            String error = null;
            try {
                HttpURLConnection h = (HttpURLConnection) new URL(r.apkUrl).openConnection();
                h.setConnectTimeout(15000);
                h.setReadTimeout(30000);
                long total = h.getContentLengthLong() > 0 ? h.getContentLengthLong() : r.apkSize, done = 0;
                try (InputStream in = h.getInputStream(); OutputStream out = new FileOutputStream(apk)) {
                    byte[] buf = new byte[1 << 16];
                    int n;
                    long lastUi = 0;
                    while ((n = in.read(buf)) > 0) {
                        if (cancelled[0]) throw new Exception("cancelled");
                        out.write(buf, 0, n);
                        done += n;
                        long now = System.currentTimeMillis();
                        if (now - lastUi > 200) {
                            lastUi = now;
                            final long d = done, t = total;
                            ui.post(() -> {
                                if (t > 0) bar.setProgress((int) (d * 1000 / t));
                                label.setText((d >> 20) + " / " + (t >> 20) + " MB");
                            });
                        }
                    }
                }
                if (total > 0 && done != total) throw new Exception("the download was incomplete");
            } catch (Exception e) {
                error = e.getMessage();
            }
            final String fe = error;
            ui.post(() -> {
                dialog.dismiss();
                if (fe != null) {
                    if (!cancelled[0]) {
                        new AlertDialog.Builder(a, android.R.style.Theme_DeviceDefault_Dialog_Alert)
                                .setMessage("The update could not be downloaded: " + fe).setPositiveButton("OK", null).show();
                    }
                    return;
                }
                install(a, apk);
            });
        }).start();
    }

    /** Hands the .apk to Android's package installer; Android shows its own confirmation. */
    private static void install(Activity a, File apk) {
        try {
            PackageInstaller pi = a.getPackageManager().getPackageInstaller();
            PackageInstaller.SessionParams p = new PackageInstaller.SessionParams(PackageInstaller.SessionParams.MODE_FULL_INSTALL);
            p.setAppPackageName(a.getPackageName());
            int id = pi.createSession(p);
            try (PackageInstaller.Session s = pi.openSession(id)) {
                try (InputStream in = new FileInputStream(apk); OutputStream out = s.openWrite("update.apk", 0, apk.length())) {
                    byte[] buf = new byte[1 << 16];
                    int n;
                    while ((n = in.read(buf)) > 0) out.write(buf, 0, n);
                    s.fsync(out);
                }
                Intent back = new Intent(a, LauncherActivity.class).setAction(ACTION_INSTALL_STATUS)
                        .addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP);
                PendingIntent pend = PendingIntent.getActivity(a, id, back,
                        PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_MUTABLE);
                s.commit(pend.getIntentSender());
            }
        } catch (Exception e) {
            new AlertDialog.Builder(a, android.R.style.Theme_DeviceDefault_Dialog_Alert)
                    .setMessage("The update could not be installed: " + e.getMessage()).setPositiveButton("OK", null).show();
        }
    }

    static final String ACTION_INSTALL_STATUS = "com.toni.wwe13.INSTALL_STATUS";

    /** LauncherActivity.onNewIntent: Android asks for the player's confirmation through this intent. */
    static boolean handleStatus(Activity a, Intent intent) {
        if (intent == null || !ACTION_INSTALL_STATUS.equals(intent.getAction())) return false;
        int status = intent.getIntExtra(PackageInstaller.EXTRA_STATUS, PackageInstaller.STATUS_FAILURE);
        if (status == PackageInstaller.STATUS_PENDING_USER_ACTION) {
            Intent confirm = intent.getParcelableExtra(Intent.EXTRA_INTENT, Intent.class);
            if (confirm != null) a.startActivity(confirm);
        } else if (status != PackageInstaller.STATUS_SUCCESS) {
            String msg = intent.getStringExtra(PackageInstaller.EXTRA_STATUS_MESSAGE);
            new AlertDialog.Builder(a, android.R.style.Theme_DeviceDefault_Dialog_Alert)
                    .setMessage("The update was not installed" + (msg != null ? ": " + msg : "."))
                    .setPositiveButton("OK", null).show();
        }
        return true;
    }
}
