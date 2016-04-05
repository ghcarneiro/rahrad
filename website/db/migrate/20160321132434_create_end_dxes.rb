class CreateEndDxes < ActiveRecord::Migration
  def change
    create_table :end_dxes do |t|
      t.string :name  # eg. Gastrointestinal
      t.string :category 
      t.integer :frequency
      t.references :dxable, polymorphic: true, index: true
      t.timestamps null: false
    end
  end
end
