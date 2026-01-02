# Migration: Before and After

## Before Migration

### Repository Structure
```
barberb's home directory:
/home/barberb/
├── generative-protein-binder-design/  (3.6+ TB)
│   ├── src/
│   ├── scripts/
│   ├── tools/
│   ├── .git/
│   └── ... (all project files)
```

### Cache Structure
```
barberb's cache directory:
/home/barberb/.cache/
├── alphafold/  (2.9 TB)
│   ├── databases/
│   ├── mgnify/
│   ├── mmseqs2/
│   ├── msa_cache/
│   ├── params/
│   ├── pdb70/
│   ├── pdb_mmcif/
│   ├── pdb_seqres/
│   ├── small_bfd/
│   ├── uniprot/
│   ├── uniref30/
│   ├── uniref90/
│   └── (6.5+ TB total)
├── typescript/
├── electron/
├── conda/
└── ... (other caches)
```

### Issues with Previous Setup
- ❌ Repository tied to single user (barberb)
- ❌ Cache only accessible from barberb's account
- ❌ Difficult for maltese to use project
- ❌ Large data in user's home directory (capacity issues)
- ❌ No formal group access control
- ❌ Unclear permissions model
- ❌ Hard to maintain and backup

---

## After Migration

### Repository Structure
```
/opt/
├── generative-protein-binder-design/  (3.6+ TB)
│   ├── src/
│   ├── scripts/
│   ├── tools/
│   ├── .git/
│   ├── MIGRATION_GUIDE.md             [NEW]
│   ├── MIGRATION_SUMMARY.md           [NEW]
│   ├── ADMIN_REFERENCE.md             [NEW]
│   ├── BEFORE_AND_AFTER.md            [NEW]
│   └── ... (all project files)
```

### Cache Structure
```
/var/cache/
├── generative-protein-binder-design/  (2.9+ TB)
│   ├── databases/
│   ├── mgnify/
│   ├── mmseqs2/
│   ├── msa_cache/
│   ├── params/
│   ├── pdb70/
│   ├── pdb_mmcif/
│   ├── pdb_seqres/
│   ├── small_bfd/
│   ├── uniprot/
│   ├── uniref30/
│   ├── uniref90/
│   ├── README.md                      [NEW]
│   └── ... (properly organized)
```

### Benefits of New Setup
- ✅ Shared access via group membership (protein-design)
- ✅ Both barberb and maltese can access project
- ✅ Easy to add new users
- ✅ Data in proper system locations (/opt, /var/cache)
- ✅ Professional permission model (rwxrwsr-x, 2770)
- ✅ Group-writable with setgid bit for consistency
- ✅ Transparent cache access via symlinks
- ✅ Easy to maintain and backup

---

## Access Comparison

### Before
```
barberb:
  cd /home/barberb/generative-protein-binder-design

maltese:
  Cannot easily access (would need to copy 6.5TB)
```

### After
```
barberb & maltese:
  cd /opt/generative-protein-binder-design
  ls ~/.cache/alphafold
```

---
Migration Date: January 2, 2026
