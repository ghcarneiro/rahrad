class CreateEndDxes < ActiveRecord::Migration
  def change
    create_table :end_dxes do |t|
      t.string :name
      t.string :category
      t.references :dx_level1, index: true, foreign_key: true # eg. abdo

      t.float :total # eg. 84.7% completed
      t.float :good # eg. 65.7% of reports with correct diagnosis
      t.float :excellent # eg. 50.6% of reports excellent
      t.integer :number # eg. 587 reports completed

      t.timestamps null: false
    end
  end
end
