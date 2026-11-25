package com.toni.wwe13;

/** JNI bridge to the shared WWE '13 launcher core (launcher2/core), see libwwe13launcher.so. */
final class LauncherNative {
    static {
        System.loadLibrary("wwe13launcher");
    }

    private LauncherNative() {}

    static native void init(String root);
    /** "base|tu|dlcInstalled|dlcKnown\ndetail" with states found / missing / wrong / corrupt. */
    static native String status();
    /** "done|total|label" of the running long operation. */
    static native String progress();
    static native void cancel();

    /** All mutators return "" on success or a player-readable error. */
    static native String extractIso(String path);
    /** "tu|name", "dlc|name" or "unknown|". */
    static native String inspectPackage(String path);
    static native String importPackage(String path);
    static native String importCreations(String[] paths, String packName, boolean replace);
    /** One line per creation: "kind|display name|package name". */
    static native String listCreations();
    static native String removeCreation(String packageName);
    /** One line per song: "path|title|format|seconds". */
    static native String listSongs();
    static native String removeSong(String path);
    static native int getLanguage();
    static native String setLanguage(int id);
    static native String backupSaves();
    /** One line per backup: "path|when|bytes|auto". */
    static native String listBackups();
    static native String backupSavesAuto();
    static native int deleteOldBackups(int keep);
    static native String restoreBackup(String path);
    /** Packs saves and creations into a backup-format zip (same format as the PC launcher's backups). */
    static native String exportSaves(String path);
    /** Changes whenever a save file changes; "" when there are no saves. */
    static native String savesFingerprint();
}
