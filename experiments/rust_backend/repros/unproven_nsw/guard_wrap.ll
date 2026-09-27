; Stores value at base[row * 65536 + col] when 0 <= offset < len.
define void @store(ptr %base, i32 range(i32 0, 65536) %row, i32 range(i32 0, 65536) %col, i32 %len, float %value) {
entry:
  %scaled = mul i32 %row, 65536
  %offset = add i32 %scaled, %col
  %nonneg = icmp sge i32 %offset, 0
  %below = icmp slt i32 %offset, %len
  %inside = and i1 %nonneg, %below
  br i1 %inside, label %write, label %done

write:
  %wide = sext i32 %offset to i64
  %address = getelementptr inbounds float, ptr %base, i64 %wide
  store float %value, ptr %address
  br label %done

done:
  ret void
}
