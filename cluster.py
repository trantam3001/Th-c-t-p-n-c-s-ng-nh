import math
import matplotlib.pyplot as plt

# =====================================================
# THAM SỐ THUẬT TOÁN SDPC
# =====================================================
MIN_DIEM = 4        # MinPts: số điểm tối thiểu để 1 điểm được xem là CORE (DBSCAN)
K_LANG_GIENG = 6   # K: số láng giềng gần nhất dùng để tính mật độ p_i


# =====================================================
# HÀM ĐỌC DỮ LIỆU TỪ FILE TXT
# =====================================================
def doc_du_lieu_tu_file(filename):
    nhan_diem = []    # danh sách nhãn điểm (A, B, C, ...)
    toa_do = []       # danh sách tọa độ (x, y)

    # Mở file dữ liệu
    with open(filename, "r", encoding="utf-8") as f:
        f.readline()  # bỏ dòng tiêu đề đầu tiên
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue  # bỏ qua dòng không hợp lệ
            nhan_diem.append(parts[0])                 # lưu nhãn điểm
            toa_do.append([float(parts[1]), float(parts[2])])  # lưu tọa độ

    # Trả về nhãn + tọa độ
    return nhan_diem, toa_do


# =====================================================
# HÀM TÍNH KHOẢNG CÁCH EUCLIDE GIỮA 2 ĐIỂM
# =====================================================
def khoang_cach(X, i, j):
    # Công thức khoảng cách Euclid trong không gian 2D
    return math.sqrt((X[i][0] - X[j][0])**2 + (X[i][1] - X[j][1])**2)


# =====================================================
# TÍNH d_max – KHOẢNG CÁCH LỚN NHẤT TRONG TẬP DỮ LIỆU
# =====================================================
def tinh_dmax(X, N):
    # dmax dùng để chuẩn hóa mật độ p_i
    dmax = 0
    for i in range(N):
        for j in range(i + 1, N):
            # lấy khoảng cách lớn nhất giữa mọi cặp điểm
            dmax = max(dmax, khoang_cach(X, i, j))
    return dmax


# =====================================================
# BƯỚC 2 – TÍNH MẬT ĐỘ p_i
# =====================================================
def tinh_mat_do_p_i(X, chi_so, dmax, N):
    # Tính khoảng cách từ điểm i (chi_so) đến tất cả điểm còn lại
    danh_sach_kc = [
        khoang_cach(X, chi_so, j)
        for j in range(N)
        if j != chi_so     # loại bỏ khoảng cách đến chính nó
    ]

    # Sắp xếp khoảng cách tăng dần
    danh_sach_kc.sort()

    # Lấy K láng giềng gần nhất
    k = min(K_LANG_GIENG, len(danh_sach_kc))

    # Công thức SDPC:
    # p_i = dmax / (trung bình K khoảng cách nhỏ nhất)
    return dmax / (sum(danh_sach_kc[:k]) / k)


# =====================================================
# BƯỚC 4 – TÍNH EPS RIÊNG CHO TÂM CỤM
# =====================================================
def tinh_eps_cho_tam(X, tam, kappa, N):
    # Tính khoảng cách từ tâm đến các điểm khác
    danh_sach_kc = [khoang_cach(X, tam, j) for j in range(N) if j != tam]
    danh_sach_kc.sort()

    # Đảm bảo kappa >= MIN_DIEM + 1
    kappa = max(MIN_DIEM + 1, min(kappa, len(danh_sach_kc)))

    # eps là khoảng cách đến láng giềng thứ kappa
    return danh_sach_kc[kappa - 1]


# =====================================================
# BƯỚC 4 – DBSCAN TẠO CỤM BAN ĐẦU
# =====================================================
def dbscan_cum_ban_dau(tam, eps, X, tap_chua_gan, N, nhan_diem):
    tap_trong_cum = [0] * N   # Sic: đánh dấu điểm thuộc cụm
    da_xet = [0] * N          # đánh dấu điểm đã xét
    hang_doi = []             # hàng đợi mở rộng DBSCAN

    print("\n[BƯỚC 4] TẠO CỤM BAN ĐẦU (DBSCAN)")

    # Khởi tạo từ tâm cụm
    tap_trong_cum[tam] = 1
    tap_chua_gan[tam] = 0
    hang_doi.append(tam)

    buoc = 0
    qh = 0

    # Mở rộng cụm theo DBSCAN
    while qh < len(hang_doi):
        p = hang_doi[qh]
        qh += 1

        if da_xet[p]:
            continue
        da_xet[p] = 1
        buoc += 1

        # Tìm hàng xóm của p trong eps
        hang_xom = [
            j for j in range(N)
            if khoang_cach(X, p, j) <= eps and j != p
        ]

        print(f"\n[Mở rộng bằng DNSCAN] xét điểm {nhan_diem[p]} (bước {buoc})")
        print("   Hàng xóm =", ", ".join(nhan_diem[x] for x in hang_xom),
              "| SL =", len(hang_xom))

        # Kiểm tra CORE
        if len(hang_xom) >= MIN_DIEM:
            print("   -> CORE, thêm hàng xóm...")
            for q in hang_xom:
                if tap_chua_gan[q]:
                    tap_trong_cum[q] = 1
                    tap_chua_gan[q] = 0
                    hang_doi.append(q)
                    print(f"      + Thêm {nhan_diem[q]}")
        else:
            print("   -> KHÔNG CORE")

    return tap_trong_cum, tap_chua_gan


# =====================================================
# TÌM SIÊU CẤP (ĐIỂM CÓ p_i LỚN HƠN GẦN NHẤT)
# =====================================================
def tim_sieu_cap(X, p, p_i, N):
    best = 1e18
    chon = -1

    # Tìm điểm có p_j > p_p và gần p nhất
    for j in range(N):
        if p_i[j] > p_i[p]:
            d = khoang_cach(X, p, j)
            if d < best:
                best = d
                chon = j

    return chon


# =====================================================
# TÌM LÁNG GIỀNG TRỰC TIẾP CHƯA THUỘC CỤM
# =====================================================
def lang_gieng_truc_tiep(X, p, tap_chua_gan, N):
    best = 1e18
    chon = -1

    # Tìm điểm chưa thuộc cụm và gần p nhất
    for j in range(N):
        if tap_chua_gan[j]:
            d = khoang_cach(X, p, j)
            if d < best:
                best = d
                chon = j

    return chon


# =====================================================
# BƯỚC 5 – MỞ RỘNG CỤM SDPC
# =====================================================
def mo_rong_cum_sdpc(tap_trong_cum, tap_chua_gan, p_i, X, N, nhan_diem):
    print("\n[BƯỚC 5] MỞ RỘNG CỤM SDPC")

    thay_doi = True
    lan = 0

    # Lặp cho đến khi không thêm được điểm mới
    while thay_doi:
        thay_doi = False
        lan += 1
        print(f"\n  Lần mở rộng {lan}")

        tap_ung_vien = set()

        # Tìm ứng viên từ các điểm đã thuộc cụm
        for i in range(N):
            if tap_trong_cum[i]:
                dn = lang_gieng_truc_tiep(X, i, tap_chua_gan, N)
                if dn != -1:
                    tap_ung_vien.add(dn)

        if not tap_ung_vien:
            print("  Không còn ứng viên")
            return

        print("  Ứng viên:", ", ".join(nhan_diem[x] for x in tap_ung_vien))

        # Kiểm tra điều kiện siêu cấp
        for p in tap_ung_vien:
            sup = tim_sieu_cap(X, p, p_i, N)
            print(f"   Xét {nhan_diem[p]} → siêu cấp = {nhan_diem[sup] if sup != -1 else '?'}")

            if sup != -1 and tap_trong_cum[sup]:
                tap_trong_cum[p] = 1
                tap_chua_gan[p] = 0
                thay_doi = True
                print(f"      + Thêm {nhan_diem[p]}")
            else:
                print("      - Không thêm")


# =====================================================
# HÀM MAIN – ĐIỀU KHIỂN TOÀN BỘ CHƯƠNG TRÌNH
# =====================================================
def main():
    nhan_diem, X = doc_du_lieu_tu_file("Acute1_4_CanThoInput.txt")
    N = len(X)

    # Tính dmax và p_i cho toàn bộ điểm
    dmax = tinh_dmax(X, N)
    p_i = [tinh_mat_do_p_i(X, i, dmax, N) for i in range(N)]

    print("\nGiá trị p_i:")
    for i in range(N):
        print(f" {nhan_diem[i]}: {p_i[i]:.6f}")

    kappa = int(input("\nNhập kappa (>4): "))

    da_phan_cum = [0] * N   # đánh dấu điểm đã được gán cụm
    ma_cum = [-1] * N       # lưu ID cụm
    so_cum = 0

    # Lặp cho đến khi hết điểm chưa phân cụm
    while True:
        tam = -1
        best = -1

        # Chọn điểm có p_i lớn nhất làm tâm
        for i in range(N):
            if not da_phan_cum[i] and p_i[i] > best:
                best = p_i[i]
                tam = i

        if tam == -1:
            break

        so_cum += 1
        print(f"\n=== CỤM {so_cum} | Tâm = {nhan_diem[tam]} ===")

        eps = tinh_eps_cho_tam(X, tam, kappa, N)
        print(f"Eps = {eps}")

        tap_chua_gan = [1 - da_phan_cum[i] for i in range(N)]

        tap_trong_cum, tap_chua_gan = dbscan_cum_ban_dau(
            tam, eps, X, tap_chua_gan, N, nhan_diem
        )

        mo_rong_cum_sdpc(tap_trong_cum, tap_chua_gan, p_i, X, N, nhan_diem)

        print("\n→ Kết quả cụm:", end=" ")
        for i in range(N):
            if tap_trong_cum[i]:
                da_phan_cum[i] = 1
                ma_cum[i] = so_cum
                print(nhan_diem[i], end=" ")
        print()

    # =====================================================
    # VẼ KẾT QUẢ PHÂN CỤM
    # =====================================================
    print("\n=== VẼ KẾT QUẢ PHÂN CỤM ===")

    mau = ["red", "blue", "green", "orange", "purple",
           "brown", "pink", "cyan", "magenta", "yellow"]

    plt.figure(figsize=(10, 7))

    for cid in range(1, so_cum + 1):
        xs = [X[i][0] for i in range(N) if ma_cum[i] == cid]
        ys = [X[i][1] for i in range(N) if ma_cum[i] == cid]
        plt.scatter(xs, ys, s=40, color=mau[(cid - 1) % len(mau)],
                    label=f"Cụm {cid}")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Kết quả phân cụm SDPC")
    plt.legend()
    plt.show()

    print("\n=== HOÀN TẤT ===")


if __name__ == "__main__":
    main()
