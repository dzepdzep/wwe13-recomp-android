package com.toni.wwe13;

import android.content.Context;
import android.content.SharedPreferences;
import android.database.Cursor;
import android.net.Uri;
import android.provider.DocumentsContract;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.security.MessageDigest;

/**
 * Save sync through one zip file the player picks once, for example in Google Drive. The zip uses the
 * launcher's backup format, so the same file can also be restored by the PC launcher.
 */
final class SaveSync {
    enum Outcome { NOT_SET_UP, UP_TO_DATE, UPLOADED, DOWNLOADED, CONFLICT, NO_SAVES, FAILED }

    static final class Result {
        final Outcome outcome;
        final String message;
        Result(Outcome o, String m) { outcome = o; message = m; }
    }

    private SaveSync() {}

    static SharedPreferences prefs(Context c) {
        return c.getSharedPreferences("launcher", Context.MODE_PRIVATE);
    }

    static boolean isSetUp(Context c) {
        return !prefs(c).getString("sync_uri", "").isEmpty();
    }

    static String syncName(Context c) {
        String uri = prefs(c).getString("sync_uri", "");
        if (uri.isEmpty()) return "";
        try (Cursor cur = c.getContentResolver().query(Uri.parse(uri),
                new String[]{DocumentsContract.Document.COLUMN_DISPLAY_NAME}, null, null, null)) {
            if (cur != null && cur.moveToFirst()) return cur.getString(0);
        } catch (Exception ignored) {
        }
        return "(not reachable right now)";
    }

    /** "size|lastModified" of the sync file, or null when it can't be read. */
    private static String remoteSignature(Context c, Uri uri) {
        try (Cursor cur = c.getContentResolver().query(uri, new String[]{
                DocumentsContract.Document.COLUMN_SIZE, DocumentsContract.Document.COLUMN_LAST_MODIFIED},
                null, null, null)) {
            if (cur != null && cur.moveToFirst()) {
                long size = cur.isNull(0) ? -1 : cur.getLong(0);
                long modified = cur.isNull(1) ? -1 : cur.getLong(1);
                return size + "|" + modified;
            }
        } catch (Exception ignored) {
        }
        return null;
    }

    private static String sha256(File f) throws Exception {
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        try (InputStream in = new FileInputStream(f)) {
            byte[] buf = new byte[1 << 16];
            int n;
            while ((n = in.read(buf)) > 0) md.update(buf, 0, n);
        }
        StringBuilder sb = new StringBuilder();
        for (byte b : md.digest()) sb.append(String.format("%02x", b));
        return sb.toString();
    }

    /**
     * Brings saves and the sync file in line. resolve: null = leave a conflict alone, true = this
     * device wins, false = the sync file wins. uploadOnly skips downloading (used when leaving the game).
     */
    static synchronized Result sync(Context c, Boolean resolve, boolean uploadOnly) {
        SharedPreferences p = prefs(c);
        String uriText = p.getString("sync_uri", "");
        if (uriText.isEmpty()) return new Result(Outcome.NOT_SET_UP, "");
        Uri uri = Uri.parse(uriText);
        File tmp = new File(c.getCacheDir(), "sync");
        tmp.mkdirs();
        try {
            String localFp = LauncherNative.savesFingerprint();
            // Download the sync file and compare its content with what this device last synced.
            // (File-size/date metadata isn't reliable on every storage provider.)
            File remoteCopy = new File(tmp, "remote.zip");
            try (InputStream in = c.getContentResolver().openInputStream(uri);
                 OutputStream out = new FileOutputStream(remoteCopy)) {
                if (in == null) return new Result(Outcome.FAILED, "The sync file can't be reached right now.");
                copy(in, out);
            } catch (java.io.FileNotFoundException e) {
                return new Result(Outcome.FAILED, "The sync file can't be reached right now.");
            }
            boolean remoteEmpty = remoteCopy.length() == 0;
            String remoteSha = remoteEmpty ? "" : sha256(remoteCopy);
            boolean localChanged = !localFp.isEmpty() && !localFp.equals(p.getString("sync_local_fp", ""));
            boolean remoteChanged = !remoteEmpty && !remoteSha.equals(p.getString("sync_remote_sha", ""));

            if (remoteChanged && localChanged && resolve == null) {
                return new Result(Outcome.CONFLICT,
                        "Your saves changed on this device and in the sync file since the last sync.");
            }
            boolean download = remoteChanged && (!localChanged || Boolean.FALSE.equals(resolve));
            boolean upload = localChanged && (!remoteChanged || Boolean.TRUE.equals(resolve));
            if (download && uploadOnly) download = false;

            if (download) {
                String err = LauncherNative.restoreBackup(remoteCopy.getAbsolutePath());  // backs up first
                if (!err.isEmpty()) return new Result(Outcome.FAILED, err);
                p.edit().putString("sync_local_fp", LauncherNative.savesFingerprint())
                        .putString("sync_remote_sha", remoteSha).apply();
                return new Result(Outcome.DOWNLOADED, "Saves downloaded from the sync file.");
            }
            if (upload) {
                File zip = new File(tmp, "upload.zip");
                String err = LauncherNative.exportSaves(zip.getAbsolutePath());
                if (!err.isEmpty()) return new Result(Outcome.FAILED, err);
                try (InputStream in = new FileInputStream(zip);
                     OutputStream out = c.getContentResolver().openOutputStream(uri, "wt")) {
                    copy(in, out);
                }
                p.edit().putString("sync_local_fp", localFp)
                        .putString("sync_remote_sha", sha256(zip)).apply();
                zip.delete();
                return new Result(Outcome.UPLOADED, "Saves uploaded to the sync file.");
            }
            if (localFp.isEmpty() && remoteEmpty) return new Result(Outcome.NO_SAVES, "No saves yet.");
            return new Result(Outcome.UP_TO_DATE, "Saves are up to date.");
        } catch (Exception e) {
            return new Result(Outcome.FAILED, "Sync failed: " + e.getMessage());
        } finally {
            new File(tmp, "remote.zip").delete();
        }
    }

    static void copy(InputStream in, OutputStream out) throws Exception {
        byte[] buf = new byte[1 << 16];
        int n;
        while ((n = in.read(buf)) > 0) out.write(buf, 0, n);
    }
}
